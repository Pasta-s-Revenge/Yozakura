from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .adapters import load_frontend, resolve_adapter
from .archive import SunArchive
from .codec import dequantize_symmetric


def _module_by_name(root: nn.Module, name: str) -> nn.Module:
    cur: Any = root
    for part in name.split("."):
        cur = cur[int(part)] if part.isdigit() else getattr(cur, part)
    return cur


def apply_sun(model: nn.Module, sun_path: str) -> nn.Module:
    manifest, tensors = SunArchive.read(sun_path)
    entries = manifest.metadata.get("module_entries", {})
    with torch.no_grad():
        for module_name, rows in entries.items():
            for row in rows:
                name = str(row["name"])
                p = int(row["prototype"])
                linear = _module_by_name(model, name)
                if not isinstance(linear, nn.Linear):
                    raise TypeError(f"Expected nn.Linear at {name}")
                device, dtype = linear.weight.device, linear.weight.dtype

                def dq(prefix: str) -> torch.Tensor:
                    return dequantize_symmetric(tensors[prefix + ".q"], tensors[prefix + ".scale"], device=device, dtype=dtype)

                left = dq(f"prototypes/{module_name}/{p}/left") + dq(f"layers/{name}/left_residual")
                right = dq(f"prototypes/{module_name}/{p}/right") + dq(f"layers/{name}/right_residual")
                linear.weight.add_(left @ right)
    return model


def load_sun_model(sun_path: str, *, device: str = "cpu", dtype: torch.dtype | None = None, trust_remote_code: bool = False, **model_kwargs: Any):
    manifest, _ = SunArchive.read(sun_path)
    if dtype is None:
        dtype = torch.float32 if device == "cpu" else torch.float16
    task = str(manifest.metadata.get("task", "causal-lm"))
    adapter, _ = resolve_adapter(manifest.base_model, task, trust_remote_code=trust_remote_code)
    model = adapter.model_class.from_pretrained(manifest.base_model, torch_dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=trust_remote_code, **model_kwargs).to(device).eval()
    apply_sun(model, sun_path)
    frontend = load_frontend(adapter, manifest.target_model, trust_remote_code=trust_remote_code)
    return model, frontend
