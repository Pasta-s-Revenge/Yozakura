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


def apply_sun(model: nn.Module, sun_path: str, *, verify_archive: bool = True) -> nn.Module:
    with SunArchive.open_tensors(sun_path, verify=verify_archive) as (manifest, tensors):
        entries = manifest.metadata.get("module_entries", {})
        with torch.no_grad():
            for module_name, rows in entries.items():
                for row in rows:
                    name = str(row["name"])
                    p = int(row["prototype"])
                    projection = _module_by_name(model, name)
                    weight = getattr(projection, "weight", None)
                    if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
                        raise TypeError(f"Expected a module with a 2D weight at {name}")
                    device, dtype = weight.device, weight.dtype

                    def dq(prefix: str) -> torch.Tensor:
                        return dequantize_symmetric(
                            tensors.get_tensor(prefix + ".q"),
                            tensors.get_tensor(prefix + ".scale"),
                            device=device,
                            dtype=dtype,
                        )

                    left = dq(f"prototypes/{module_name}/{p}/left")
                    left.add_(dq(f"layers/{name}/left_residual"))
                    right = dq(f"prototypes/{module_name}/{p}/right")
                    right.add_(dq(f"layers/{name}/right_residual"))
                    if (left.shape[0], right.shape[1]) != tuple(weight.shape):
                        raise ValueError(
                            f"Reconstructed delta shape mismatch at {name}: "
                            f"{(left.shape[0], right.shape[1])} != {tuple(weight.shape)}"
                        )
                    weight.addmm_(left, right)
                    del left, right
    return model


def load_sun_model(
    sun_path: str,
    *,
    device: str = "cpu",
    dtype: torch.dtype | None = None,
    trust_remote_code: bool = False,
    verify_archive: bool = True,
    **model_kwargs: Any,
):
    manifest = SunArchive.read_manifest(sun_path)
    if dtype is None:
        dtype = torch.float16
    task = str(manifest.metadata.get("task", "causal-lm"))
    adapter, _ = resolve_adapter(manifest.base_model, task, trust_remote_code=trust_remote_code)
    model = adapter.model_class.from_pretrained(
        manifest.base_model,
        dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=trust_remote_code,
        **model_kwargs,
    ).to(device).eval()
    apply_sun(model, sun_path, verify_archive=verify_archive)
    frontend = load_frontend(adapter, manifest.target_model, trust_remote_code=trust_remote_code)
    return model, frontend
