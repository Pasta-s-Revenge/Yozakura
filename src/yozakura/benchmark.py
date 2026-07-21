from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from .adapters import load_frontend, resolve_adapter
from .archive import SunArchive
from .runtime import load_sun_model


def _sync(device: str) -> None:
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def _linux_memory_kib(field: str) -> int | None:
    status = Path("/proc/self/status")
    if not status.exists():
        return None
    for line in status.read_text().splitlines():
        if line.startswith(field + ":"):
            return int(line.split()[1])
    return None


def _dtype(name: str, device: str) -> torch.dtype:
    if name == "auto":
        return torch.float16 if device == "cuda" else torch.float32
    mapping = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    return mapping[name]


def _move_inputs(inputs: Any, device: str) -> Any:
    if hasattr(inputs, "to"):
        return inputs.to(device)
    if isinstance(inputs, dict):
        return {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
    return inputs


def _token_count(inputs: Any) -> int:
    ids = inputs.get("input_ids") if isinstance(inputs, dict) else getattr(inputs, "input_ids", None)
    if ids is None:
        raise RuntimeError("Benchmark currently requires a text frontend with input_ids")
    return int(ids.shape[-1])


def _decode(frontend: Any, token_ids: torch.Tensor) -> str:
    if hasattr(frontend, "decode"):
        return frontend.decode(token_ids, skip_special_tokens=True)
    if hasattr(frontend, "batch_decode"):
        return frontend.batch_decode(token_ids[None, :], skip_special_tokens=True)[0]
    return ""


def _model_bytes(model: torch.nn.Module) -> int:
    return sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())


def _prompt_quality(model: torch.nn.Module, inputs: Any) -> dict[str, Any]:
    ids = inputs.get("input_ids") if isinstance(inputs, dict) else getattr(inputs, "input_ids", None)
    if ids is None or ids.shape[-1] < 2:
        return {"prompt_loss": None, "prompt_perplexity": None, "next_token_top10": []}

    with torch.inference_mode():
        output = model(**inputs)
        logits = getattr(output, "logits", None)
        if logits is None:
            return {"prompt_loss": None, "prompt_perplexity": None, "next_token_top10": []}
        shift_logits = logits[:, :-1, :].float()
        shift_labels = ids[:, 1:].to(shift_logits.device)
        loss = torch.nn.functional.cross_entropy(
            shift_logits.reshape(-1, shift_logits.shape[-1]),
            shift_labels.reshape(-1),
        )
        next_logits = logits[0, -1].float()
        values, indices = torch.topk(next_logits, k=min(10, next_logits.numel()))

    loss_value = float(loss.cpu())
    return {
        "prompt_loss": loss_value,
        "prompt_perplexity": math.exp(min(loss_value, 20.0)),
        "next_token_top10": [
            {"token_id": int(token_id), "logit": float(value)}
            for token_id, value in zip(indices.cpu(), values.cpu())
        ],
    }


def _load_worker_model(args: argparse.Namespace):
    dtype = _dtype(args.dtype, args.device)
    manifest, _ = SunArchive.read(args.archive)
    if args.worker_mode == "sun":
        return load_sun_model(
            args.archive,
            device=args.device,
            dtype=dtype,
            trust_remote_code=args.trust_remote_code,
        )

    task = str(manifest.metadata.get("task", "auto"))
    adapter, _ = resolve_adapter(
        manifest.target_model,
        task,
        trust_remote_code=args.trust_remote_code,
    )
    model = adapter.model_class.from_pretrained(
        manifest.target_model,
        dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=args.trust_remote_code,
    ).to(args.device).eval()
    frontend = load_frontend(
        adapter,
        manifest.target_model,
        trust_remote_code=args.trust_remote_code,
    )
    return model, frontend


def _worker(args: argparse.Namespace) -> dict[str, Any]:
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    gc.collect()
    if args.device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    rss_before = _linux_memory_kib("VmRSS")
    started = time.perf_counter()
    model, frontend = _load_worker_model(args)
    _sync(args.device)
    load_seconds = time.perf_counter() - started

    inputs = _move_inputs(frontend(args.prompt, return_tensors="pt"), args.device)
    prompt_tokens = _token_count(inputs)
    quality = _prompt_quality(model, inputs)

    with torch.inference_mode():
        _sync(args.device)
        first_started = time.perf_counter()
        model.generate(**inputs, max_new_tokens=1, do_sample=False)
        _sync(args.device)
        first_token_seconds = time.perf_counter() - first_started

        for _ in range(args.warmup):
            model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        _sync(args.device)

        durations: list[float] = []
        generated_counts: list[int] = []
        first_output: torch.Tensor | None = None
        for _ in range(args.runs):
            _sync(args.device)
            run_started = time.perf_counter()
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )
            _sync(args.device)
            durations.append(time.perf_counter() - run_started)
            generated_counts.append(max(int(output.shape[-1]) - prompt_tokens, 0))
            if first_output is None:
                first_output = output[0].detach().cpu()

    assert first_output is not None
    generated_ids = first_output[prompt_tokens:]
    total_generated = sum(generated_counts)
    total_seconds = sum(durations)

    return {
        "label": args.worker_mode,
        "device": args.device,
        "dtype": args.dtype,
        "load_seconds": load_seconds,
        "first_token_seconds": first_token_seconds,
        "run_seconds": durations,
        "latency_mean_seconds": statistics.mean(durations),
        "latency_median_seconds": statistics.median(durations),
        "generated_tokens": generated_counts,
        "tokens_per_second": total_generated / total_seconds if total_seconds else None,
        "prompt_tokens": prompt_tokens,
        "model_parameter_bytes": _model_bytes(model),
        "rss_before_kib": rss_before,
        "rss_after_load_kib": _linux_memory_kib("VmRSS"),
        "peak_rss_kib": _linux_memory_kib("VmHWM"),
        "peak_cuda_bytes": torch.cuda.max_memory_allocated() if args.device == "cuda" else None,
        "output_token_ids": generated_ids.tolist(),
        "output_text": _decode(frontend, generated_ids),
        **quality,
    }


def _run_isolated(args: argparse.Namespace, mode: str) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "yozakura.benchmark",
        "--worker-mode",
        mode,
        "--archive",
        args.archive,
        "--prompt",
        args.prompt,
        "--device",
        args.device,
        "--dtype",
        args.dtype,
        "--runs",
        str(args.runs),
        "--warmup",
        str(args.warmup),
        "--max-new-tokens",
        str(args.max_new_tokens),
    ]
    if args.trust_remote_code:
        command.append("--trust-remote-code")
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    return json.loads(completed.stdout)


def _agreement(left: list[int], right: list[int]) -> float | None:
    compared = min(len(left), len(right))
    if compared == 0:
        return None
    return sum(a == b for a, b in zip(left[:compared], right[:compared])) / compared


def _ratio(numerator: float | int | None, denominator: float | int | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return float(numerator) / float(denominator)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark a Yozakura .sun archive against its target model")
    parser.add_argument("--archive", required=True)
    parser.add_argument("--prompt", default="The future of open source AI is")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["auto", "fp16", "bf16", "fp32"], default="auto")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--compare-target", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--output", help="Optional JSON report path")
    parser.add_argument("--worker-mode", choices=["sun", "target"], help=argparse.SUPPRESS)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.runs < 1 or args.warmup < 0 or args.max_new_tokens < 1:
        raise SystemExit("runs and max-new-tokens must be positive; warmup must be non-negative")

    if args.worker_mode:
        print(json.dumps(_worker(args), ensure_ascii=False))
        return

    archive = Path(args.archive)
    manifest, _ = SunArchive.read(archive)
    sun = _run_isolated(args, "sun")
    target = _run_isolated(args, "target") if args.compare_target else None

    report: dict[str, Any] = {
        "system": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "archive": {
            "path": str(archive),
            "bytes": archive.stat().st_size,
            "manifest": asdict(manifest),
        },
        "sun": sun,
        "target": target,
    }

    if target:
        sun_top = {row["token_id"] for row in sun["next_token_top10"]}
        target_top = {row["token_id"] for row in target["next_token_top10"]}
        report["comparison"] = {
            "generation_token_agreement": _agreement(sun["output_token_ids"], target["output_token_ids"]),
            "next_token_top10_overlap": len(sun_top & target_top) / max(len(sun_top | target_top), 1),
            "speed_ratio_sun_over_target": _ratio(sun["tokens_per_second"], target["tokens_per_second"]),
            "load_time_ratio_sun_over_target": _ratio(sun["load_seconds"], target["load_seconds"]),
            "peak_rss_ratio_sun_over_target": _ratio(sun["peak_rss_kib"], target["peak_rss_kib"]),
            "peak_cuda_ratio_sun_over_target": _ratio(sun["peak_cuda_bytes"], target["peak_cuda_bytes"]),
            "prompt_loss_delta": (
                sun["prompt_loss"] - target["prompt_loss"]
                if sun["prompt_loss"] is not None and target["prompt_loss"] is not None
                else None
            ),
        }

    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
