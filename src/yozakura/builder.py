from __future__ import annotations

import gc
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch

from .adapters import matching_linears, resolve_adapter
from .archive import SunArchive, SunManifest
from .codec import quantize_symmetric


@dataclass(slots=True)
class BuildConfig:
    base_model: str
    target_model: str
    output: str
    modules: tuple[str, ...] | None = None
    task: str = "auto"
    rank: int = 32
    prototypes_per_module: int = 4
    max_layers: int | None = None
    device: str = "cpu"
    dtype: torch.dtype = torch.float16
    trust_remote_code: bool = False
    max_relative_nmse: float = 0.15
    max_distribution_ratio: float = 0.75
    svd_oversample: int = 8
    error_chunk_rows: int = 256


def _factorize(delta: torch.Tensor, rank: int, oversample: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    delta = delta.float()
    r = min(rank, *delta.shape)
    if r == min(delta.shape):
        u, s, vh = torch.linalg.svd(delta, full_matrices=False)
        root = s[:r].clamp_min(0).sqrt()
        return u[:, :r] * root, root[:, None] * vh[:r, :]

    q = min(r + max(oversample, 0), min(delta.shape))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        u, s, v = torch.svd_lowrank(delta, q=q, niter=2)
    root = s[:r].clamp_min(0).sqrt()
    return u[:, :r] * root, root[:, None] * v[:, :r].T


def _residual_energy(delta: torch.Tensor, left: torch.Tensor, right: torch.Tensor, chunk_rows: int) -> float:
    total = 0.0
    rows = max(int(chunk_rows), 1)
    for start in range(0, delta.shape[0], rows):
        stop = min(start + rows, delta.shape[0])
        residual = delta[start:stop] - left[start:stop] @ right
        total += float(residual.square().sum())
        del residual
    return total


def _kmeans_rows(x: torch.Tensor, k: int, steps: int = 25) -> torch.Tensor:
    n = x.shape[0]
    k = min(k, n)
    centers = x[torch.linspace(0, n - 1, k).round().long()].clone()
    for _ in range(steps):
        labels = torch.cdist(x, centers).argmin(dim=1)
        new = torch.stack([x[labels == i].mean(0) if (labels == i).any() else centers[i] for i in range(k)])
        if torch.allclose(new, centers, atol=1e-5, rtol=1e-4):
            break
        centers = new
    return torch.cdist(x, centers).argmin(dim=1)


def build_sun(config: BuildConfig) -> Path:
    common = dict(dtype=config.dtype, low_cpu_mem_usage=True, trust_remote_code=config.trust_remote_code)
    base_adapter, base_cfg = resolve_adapter(config.base_model, config.task, trust_remote_code=config.trust_remote_code)
    target_adapter, target_cfg = resolve_adapter(config.target_model, config.task, trust_remote_code=config.trust_remote_code)
    if base_adapter.task != target_adapter.task:
        raise ValueError(f"Task mismatch: {base_adapter.task} != {target_adapter.task}")
    base = base_adapter.model_class.from_pretrained(config.base_model, **common).to(config.device).eval()
    target = target_adapter.model_class.from_pretrained(config.target_model, **common).to(config.device).eval()
    base_linears, target_linears, selected_modules = matching_linears(base, target, config.modules)
    names = sorted(base_linears)
    if config.max_layers is not None:
        names = names[: config.max_layers * max(len(selected_modules), 1)]
    if not names:
        raise RuntimeError("No repeated matching linear modules found; pass --modules explicitly")

    grouped = defaultdict(list)
    raw_target_bytes = 0
    error = energy = 0.0
    with torch.no_grad():
        for name in names:
            bw, tw = base_linears[name].weight, target_linears[name].weight
            delta = (tw - bw).to(device="cpu", dtype=torch.float32)
            left, right = _factorize(delta, config.rank, config.svd_oversample)
            grouped[name.rsplit(".", 1)[-1]].append((name, left, right))
            error += _residual_energy(delta, left, right, config.error_chunk_rows)
            energy += float(delta.square().sum())
            raw_target_bytes += tw.numel() * 2
            del delta
            if config.device == "cuda":
                torch.cuda.empty_cache()

    del base_linears, target_linears, base, target
    gc.collect()
    if config.device == "cuda":
        torch.cuda.empty_cache()

    tensors: dict[str, torch.Tensor] = {}
    module_entries = {}
    for module, items in grouped.items():
        sketches = torch.stack([torch.cat([l.mean(0), r.mean(1)]) for _, l, r in items])
        labels = _kmeans_rows(sketches, config.prototypes_per_module)
        proto_count = int(labels.max()) + 1
        prototype_l, prototype_r = [], []
        for p in range(proto_count):
            members = [items[i] for i in range(len(items)) if int(labels[i]) == p]
            prototype_l.append(torch.stack([x[1] for x in members]).mean(0))
            prototype_r.append(torch.stack([x[2] for x in members]).mean(0))
        entries = []
        for idx, (name, left, right) in enumerate(items):
            p = int(labels[idx])
            for key, value in (("left_residual", left - prototype_l[p]), ("right_residual", right - prototype_r[p])):
                q, s = quantize_symmetric(value)
                tensors[f"layers/{name}/{key}.q"] = q
                tensors[f"layers/{name}/{key}.scale"] = s
            entries.append({"name": name, "prototype": p})
        for p, (left, right) in enumerate(zip(prototype_l, prototype_r)):
            for key, value in (("left", left), ("right", right)):
                q, s = quantize_symmetric(value)
                tensors[f"prototypes/{module}/{p}/{key}.q"] = q
                tensors[f"prototypes/{module}/{p}/{key}.scale"] = s
        module_entries[module] = entries
        del items, sketches, labels, prototype_l, prototype_r

    nmse = error / max(energy, 1e-12)
    metadata = {
        "task": base_adapter.task,
        "base_model_type": getattr(base_cfg, "model_type", "unknown"),
        "target_model_type": getattr(target_cfg, "model_type", "unknown"),
        "module_entries": module_entries,
        "svd_delta_nmse": nmse,
        "raw_target_bytes": raw_target_bytes,
        "build_dtype": str(config.dtype).removeprefix("torch."),
    }
    manifest = SunManifest(base_model=config.base_model, target_model=config.target_model, modules=list(selected_modules), rank=config.rank, prototypes_per_module=config.prototypes_per_module, metadata=metadata)
    out = SunArchive.write(config.output, manifest, tensors)
    ratio = out.stat().st_size / max(raw_target_bytes, 1)
    manifest.metadata.update(artifact_bytes=out.stat().st_size, distribution_ratio=ratio)
    out = SunArchive.write(out, manifest, tensors)
    failures = []
    if nmse > config.max_relative_nmse:
        failures.append(f"svd_delta_nmse={nmse:.6f} > {config.max_relative_nmse}")
    if ratio >= config.max_distribution_ratio:
        failures.append(f"distribution_ratio={ratio:.6f} >= {config.max_distribution_ratio}")
    if failures:
        out.unlink(missing_ok=True)
        raise RuntimeError("Release gate failed: " + "; ".join(failures))
    return out
