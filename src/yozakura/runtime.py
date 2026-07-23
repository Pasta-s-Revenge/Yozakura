from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

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
    bytes_per_row = max(2 * rank * element_size, 1)
    return max(1, min(output_rows, workspace_mib * 1024 * 1024 // bytes_per_row))


def apply_sun(
    model: nn.Module,
    sun_path: str,
    *,
    verify_archive: bool = True,
    workspace_mib: int = DEFAULT_WORKSPACE_MIB,
) -> nn.Module:
    """Apply a SUN delta with bounded reconstruction workspace."""
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
                    if weight.device.type == "meta":
                        raise RuntimeError(
                            f"Cannot apply SUN delta to offloaded meta weight at {name}; "
                            "apply the archive before model dispatch"
                        )
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


def _dispatch_model(
    model: nn.Module,
    *,
    max_memory: Mapping[int | str, int | str] | None,
    offload_folder: str | None,
) -> nn.Module:
    """Dispatch a reconstructed model across GPU, CPU, and optionally disk."""
    try:
        from accelerate import dispatch_model, infer_auto_device_map
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("Tiered offload requires accelerate>=1.0") from exc

    if offload_folder is not None:
        Path(offload_folder).mkdir(parents=True, exist_ok=True)

    device_map = infer_auto_device_map(model, max_memory=dict(max_memory) if max_memory else None)
    return dispatch_model(
        model,
        device_map=device_map,
        offload_dir=offload_folder,
        offload_buffers=True,
    )


def model_input_device(model: nn.Module) -> torch.device:
    """Return the device expected by the model's input embedding layer."""
    get_embeddings = getattr(model, "get_input_embeddings", None)
    if callable(get_embeddings):
        embeddings = get_embeddings()
        weight = getattr(embeddings, "weight", None)
        if isinstance(weight, torch.Tensor) and weight.device.type != "meta":
            return weight.device
    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    return torch.device("cpu")


def load_sun_model(
    sun_path: str,
    *,
    device: str = "cpu",
    dtype: torch.dtype | None = None,
    trust_remote_code: bool = False,
    verify_archive: bool = True,
    workspace_mib: int = DEFAULT_WORKSPACE_MIB,
    max_memory: Mapping[int | str, int | str] | None = None,
    offload_folder: str | None = None,
    checkpoint_cache: str = ".yozakura-checkpoints",
    revision: str | None = None,
    local_files_only: bool = False,
    **model_kwargs: Any,
):
    """Load a SUN model using eager, tiered, or bounded-memory construction.

    ``device='out-of-core'`` reconstructs one tensor at a time and dispatches the
    model across available tiers. ``device='layer'`` additionally limits the
    default CPU-resident weights to 1 GiB so Transformer blocks are streamed
    from disk by Accelerate's forward hooks.
    """
    if dtype is None:
        dtype = torch.float16
    if device in {"out-of-core", "layer"}:
        if model_kwargs:
            unsupported = ", ".join(sorted(model_kwargs))
            raise ValueError(f"Out-of-core loading does not accept model kwargs yet: {unsupported}")
        from .out_of_core import load_out_of_core_sun_model

        return load_out_of_core_sun_model(
            sun_path,
            dtype=dtype,
            max_memory=max_memory,
            offload_folder=offload_folder,
            checkpoint_cache=checkpoint_cache,
            workspace_mib=workspace_mib,
            verify_archive=verify_archive,
            trust_remote_code=trust_remote_code,
            revision=revision,
            local_files_only=local_files_only,
            layer_streaming=device == "layer",
        )

    manifest = SunArchive.read_manifest(sun_path)
    task = str(manifest.metadata.get("task", "causal-lm"))
    adapter, _ = resolve_adapter(manifest.base_model, task, trust_remote_code=trust_remote_code)
    model = adapter.model_class.from_pretrained(
        manifest.base_model,
        dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=trust_remote_code,
        revision=revision,
        local_files_only=local_files_only,
        **model_kwargs,
    ).eval()

    if device != "auto":
        model = model.to(device)
    apply_sun(
        model,
        sun_path,
        verify_archive=verify_archive,
        workspace_mib=workspace_mib,
    )
    if device == "auto":
        model = _dispatch_model(model, max_memory=max_memory, offload_folder=offload_folder)

    frontend = load_frontend(adapter, manifest.target_model, trust_remote_code=trust_remote_code)
    return model, frontend
