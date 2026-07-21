from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .adapters import load_frontend, resolve_adapter
from .archive import SunArchive
from .codec import dequantize_symmetric


DEFAULT_WORKSPACE_MIB = 256


def _module_by_name(root: nn.Module, name: str) -> nn.Module:
    cur: Any = root
    for part in name.split("."):
        cur = cur[int(part)] if part.isdigit() else getattr(cur, part)
    return cur


def _tensor_slice(tensors: Any, name: str, start: int, stop: int) -> torch.Tensor:
    """Read a row slice without materializing the full tensor when supported."""
    if hasattr(tensors, "get_slice"):
        return tensors.get_slice(name)[start:stop]
    return tensors.get_tensor(name)[start:stop]


def _dequantized_rows(
    tensors: Any,
    prefix: str,
    start: int,
    stop: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return dequantize_symmetric(
        _tensor_slice(tensors, prefix + ".q", start, stop),
        tensors.get_tensor(prefix + ".scale"),
        device=device,
        dtype=dtype,
    )


def _workspace_rows(
    *,
    output_rows: int,
    rank: int,
    dtype: torch.dtype,
    workspace_mib: int,
) -> int:
    if workspace_mib < 1:
        raise ValueError("workspace_mib must be positive")
    if rank < 1:
        raise ValueError("rank must be positive")
    element_size = torch.empty((), dtype=dtype).element_size()
    # Two left-factor chunks coexist briefly: prototype and residual.
    bytes_per_row = max(2 * rank * element_size, 1)
    return max(1, min(output_rows, workspace_mib * 1024 * 1024 // bytes_per_row))


def apply_sun(
    model: nn.Module,
    sun_path: str,
    *,
    verify_archive: bool = True,
    workspace_mib: int = DEFAULT_WORKSPACE_MIB,
) -> nn.Module:
    """Apply a SUN delta with bounded reconstruction workspace.

    The low-rank right factor is kept resident because it is small. The larger
    left factor is decoded and multiplied in row chunks, so peak temporary
    memory is controlled by ``workspace_mib`` rather than projection size.
    """
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

                    right = dq(f"prototypes/{module_name}/{p}/right")
                    right.add_(dq(f"layers/{name}/right_residual"))
                    output_rows, input_columns = tuple(weight.shape)
                    if right.ndim != 2 or right.shape[1] != input_columns:
                        raise ValueError(
                            f"Reconstructed right-factor shape mismatch at {name}: "
                            f"{tuple(right.shape)} is incompatible with {tuple(weight.shape)}"
                        )

                    rank = int(right.shape[0])
                    chunk_rows = _workspace_rows(
                        output_rows=output_rows,
                        rank=rank,
                        dtype=dtype,
                        workspace_mib=workspace_mib,
                    )
                    prototype_prefix = f"prototypes/{module_name}/{p}/left"
                    residual_prefix = f"layers/{name}/left_residual"
                    for start in range(0, output_rows, chunk_rows):
                        stop = min(start + chunk_rows, output_rows)
                        left = _dequantized_rows(
                            tensors,
                            prototype_prefix,
                            start,
                            stop,
                            device=device,
                            dtype=dtype,
                        )
                        left.add_(
                            _dequantized_rows(
                                tensors,
                                residual_prefix,
                                start,
                                stop,
                                device=device,
                                dtype=dtype,
                            )
                        )
                        if left.ndim != 2 or left.shape[1] != rank:
                            raise ValueError(
                                f"Reconstructed left-factor shape mismatch at {name}: "
                                f"{tuple(left.shape)} is incompatible with rank {rank}"
                            )
                        weight[start:stop].addmm_(left, right)
                        del left
                    del right
    return model


def load_sun_model(
    sun_path: str,
    *,
    device: str = "cpu",
    dtype: torch.dtype | None = None,
    trust_remote_code: bool = False,
    verify_archive: bool = True,
    workspace_mib: int = DEFAULT_WORKSPACE_MIB,
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
    apply_sun(
        model,
        sun_path,
        verify_archive=verify_archive,
        workspace_mib=workspace_mib,
    )
    frontend = load_frontend(adapter, manifest.target_model, trust_remote_code=trust_remote_code)
    return model, frontend
