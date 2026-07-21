from __future__ import annotations

import argparse
import json
from dataclasses import asdict

import torch

from .adapters import ADAPTERS, resolve_adapter
from .archive import SunArchive
from .builder import BuildConfig, build_sun
from .runtime import DEFAULT_WORKSPACE_MIB, load_sun_model, model_input_device


DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="yozakura")
    sub = p.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="Build a distributable .sun hypernetwork delta")
    b.add_argument("--base", required=True)
    b.add_argument("--target", required=True)
    b.add_argument("--output", required=True)
    b.add_argument("--modules", default="auto", help="Comma-separated projection suffixes, auto, or all")
    b.add_argument("--task", default="auto", choices=["auto", *ADAPTERS])
    b.add_argument("--rank", type=int, default=32)
    b.add_argument("--prototypes", type=int, default=4)
    b.add_argument("--max-layers", type=int)
    b.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    b.add_argument("--dtype", choices=DTYPES, default="float16", help="Model load dtype; float16 minimizes RAM")
    b.add_argument("--svd-oversample", type=int, default=8)
    b.add_argument("--error-chunk-rows", type=int, default=256)
    b.add_argument("--trust-remote-code", action="store_true")
    r = sub.add_parser("run", help="Run generation from a generative .sun archive")
    r.add_argument("archive")
    r.add_argument("--prompt", required=True)
    r.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    r.add_argument("--dtype", choices=DTYPES, default="float16", help="Model dtype; float16 minimizes RAM")
    r.add_argument("--max-new-tokens", type=int, default=128)
    r.add_argument(
        "--workspace-mib",
        type=int,
        default=DEFAULT_WORKSPACE_MIB,
        help="Maximum temporary MiB used while reconstructing each projection",
    )
    r.add_argument(
        "--max-memory",
        action="append",
        default=[],
        metavar="DEVICE=LIMIT",
        help="Tier budget, e.g. 0=8GiB or cpu=24GiB; repeat per device",
    )
    r.add_argument("--offload-folder", help="Directory for disk-offloaded model modules")
    r.add_argument("--skip-checksum", action="store_true", help="Skip .sun SHA-256 verification for faster startup")
    r.add_argument("--trust-remote-code", action="store_true")
    i = sub.add_parser("inspect", help="Print the .sun manifest")
    i.add_argument("archive")
    s = sub.add_parser("probe", help="Resolve a Hugging Face model task without loading weights")
    s.add_argument("model")
    s.add_argument("--task", default="auto", choices=["auto", *ADAPTERS])
    s.add_argument("--trust-remote-code", action="store_true")
    return p


def _parse_modules(value: str) -> tuple[str, ...] | None:
    normalized = value.strip().lower()
    if normalized == "auto":
        return None
    if normalized in {"all", "*"}:
        return ("*",)
    modules = tuple(x.strip() for x in value.split(",") if x.strip())
    if not modules:
        raise SystemExit("--modules must be auto, all, or a comma-separated suffix list")
    return modules


def _parse_max_memory(values: list[str]) -> dict[int | str, str] | None:
    if not values:
        return None
    parsed: dict[int | str, str] = {}
    for value in values:
        key, separator, limit = value.partition("=")
        key, limit = key.strip(), limit.strip()
        if not separator or not key or not limit:
            raise SystemExit("--max-memory must use DEVICE=LIMIT, for example 0=8GiB")
        device: int | str = int(key) if key.isdigit() else key
        parsed[device] = limit
    return parsed


def main() -> None:
    args = _parser().parse_args()
    if args.command == "build":
        modules = _parse_modules(args.modules)
        path = build_sun(
            BuildConfig(
                base_model=args.base,
                target_model=args.target,
                output=args.output,
                modules=modules,
                task=args.task,
                rank=args.rank,
                prototypes_per_module=args.prototypes,
                max_layers=args.max_layers,
                device=args.device,
                dtype=DTYPES[args.dtype],
                trust_remote_code=args.trust_remote_code,
                svd_oversample=args.svd_oversample,
                error_chunk_rows=args.error_chunk_rows,
            )
        )
        print(path)
    elif args.command == "inspect":
        manifest = SunArchive.read_manifest(args.archive)
        print(json.dumps(asdict(manifest), ensure_ascii=False, indent=2, default=str))
    elif args.command == "probe":
        adapter, config = resolve_adapter(args.model, args.task, trust_remote_code=args.trust_remote_code)
        print(json.dumps({"model": args.model, "model_type": getattr(config, "model_type", None), "task": adapter.task, "generative": adapter.generative, "model_class": adapter.model_class.__name__}, indent=2))
    else:
        if args.workspace_mib < 1:
            raise SystemExit("--workspace-mib must be positive")
        if args.device != "auto" and (args.max_memory or args.offload_folder):
            raise SystemExit("--max-memory and --offload-folder require --device auto")
        manifest = SunArchive.read_manifest(args.archive)
        task = str(manifest.metadata.get("task", "causal-lm"))
        adapter, _ = resolve_adapter(manifest.base_model, task, trust_remote_code=args.trust_remote_code)
        if not adapter.generative:
            raise SystemExit(f"Task {task!r} is not generative; use load_sun_model() from Python")
        model, frontend = load_sun_model(
            args.archive,
            device=args.device,
            dtype=DTYPES[args.dtype],
            trust_remote_code=args.trust_remote_code,
            verify_archive=not args.skip_checksum,
            workspace_mib=args.workspace_mib,
            max_memory=_parse_max_memory(args.max_memory),
            offload_folder=args.offload_folder,
        )
        inputs = frontend(args.prompt, return_tensors="pt").to(model_input_device(model))
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
        print(frontend.decode(output[0], skip_special_tokens=True) if hasattr(frontend, "decode") else output)


if __name__ == "__main__":
    main()
