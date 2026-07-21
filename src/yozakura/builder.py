from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from .archive import SunArchive, SunManifest
from .codec import quantize_symmetric


@dataclass(slots=True)
class BuildConfig:
    base_model: str
    target_model: str
    output: str
    modules: tuple[str, ...] = ("gate_proj", "up_proj", "down_proj")
    rank: int = 32
    prototypes_per_module: int = 4
    max_layers: int | None = None
    device: str = "cpu"
    trust_remote_code: bool = False
    max_relative_nmse: float = 0.15
    max_distribution_ratio: float = 0.75


def _named_linears(model: nn.Module, suffixes: tuple[str, ...]) -> dict[str, nn.Linear]:
    return {name: module for name, module in model.named_modules() if isinstance(module, nn.Linear) and name.rsplit(".", 1)[-1] in suffixes}


def _factorize(delta: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    delta = delta.float()
    r = min(rank, *delta.shape)
    u, s, vh = torch.linalg.svd(delta, full_matrices=False)
    root = s[:r].clamp_min(0).sqrt()
    return u[:, :r] * root, root[:, None] * vh[:r, :]


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
    dtype = torch.float32 if config.device == "cpu" else torch.float16
    common = dict(torch_dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=config.trust_remote_code)
    base = AutoModelForCausalLM.from_pretrained(config.base_model, **common).to(config.device).eval()
    target = AutoModelForCausalLM.from_pretrained(config.target_model, **common).to(config.device).eval()
    base_linears = _named_linears(base, config.modules)
    target_linears = _named_linears(target, config.modules)
    names = sorted(set(base_linears) & set(target_linears))
    if config.max_layers is not None:
        names = names[: config.max_layers * len(config.modules)]
    if not names:
        raise RuntimeError("No matching target linear modules found")

    grouped = defaultdict(list)
    raw_target_bytes = 0
    error = energy = 0.0
    with torch.no_grad():
        for name in names:
            bw, tw = base_linears[name].weight, target_linears[name].weight
            if bw.shape != tw.shape:
                raise ValueError(f"Shape mismatch for {name}: {bw.shape} != {tw.shape}")
            delta = (tw - bw).cpu().float()
            left, right = _factorize(delta, config.rank)
            grouped[name.rsplit(".", 1)[-1]].append((name, left, right))
            approx = left @ right
            error += float((delta - approx).square().sum())
            energy += float(delta.square().sum())
            raw_target_bytes += tw.numel() * 2

    tensors = {}
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

    nmse = error / max(energy, 1e-12)
    manifest = SunManifest(base_model=config.base_model, target_model=config.target_model, modules=list(config.modules), rank=config.rank, prototypes_per_module=config.prototypes_per_module, metadata={"module_entries": module_entries, "svd_delta_nmse": nmse, "raw_target_bytes": raw_target_bytes})
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
