from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors.torch import save_file

from .adapters import load_frontend, resolve_adapter
from .archive import SunArchive
from .checkpoint import ShardedCheckpoint
from .codec import dequantize_symmetric


def _workspace_rows(*, output_rows: int, rank: int, dtype: torch.dtype, workspace_mib: int) -> int:
    if workspace_mib < 1:
        raise ValueError("workspace_mib must be positive")
    if rank < 1:
        raise ValueError("rank must be positive")
    element_size = torch.empty((), dtype=dtype).element_size()
    bytes_per_row = max(2 * rank * element_size, 1)
    return max(1, min(output_rows, workspace_mib * 1024 * 1024 // bytes_per_row))


def _tensor_slice(tensors: Any, name: str, start: int, stop: int) -> torch.Tensor:
    if hasattr(tensors, "get_slice"):
        return tensors.get_slice(name)[start:stop]
    return tensors.get_tensor(name)[start:stop]


def _dequantized_rows(
    tensors: Any,
    prefix: str,
    start: int,
    stop: int,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    return dequantize_symmetric(
        _tensor_slice(tensors, prefix + ".q", start, stop),
        tensors.get_tensor(prefix + ".scale"),
        device=torch.device("cpu"),
        dtype=dtype,
    )


def _patch_projection_weight(
    weight: torch.Tensor,
    *,
    module_name: str,
    module_type: str,
    prototype: int,
    tensors: Any,
    workspace_mib: int,
) -> torch.Tensor:
    if weight.ndim != 2:
        raise ValueError(f"SUN projection weight must be 2D: {module_name}")
    if not weight.is_floating_point():
        raise TypeError(f"SUN projection weight must be floating point: {module_name}")

    dtype = weight.dtype

    def dq(prefix: str) -> torch.Tensor:
        return dequantize_symmetric(
            tensors.get_tensor(prefix + ".q"),
            tensors.get_tensor(prefix + ".scale"),
            device=torch.device("cpu"),
            dtype=dtype,
        )

    right = dq(f"prototypes/{module_type}/{prototype}/right")
    right.add_(dq(f"layers/{module_name}/right_residual"))
    output_rows, input_columns = tuple(weight.shape)
    if right.ndim != 2 or right.shape[1] != input_columns:
        raise ValueError(
            f"Reconstructed right-factor shape mismatch at {module_name}: "
            f"{tuple(right.shape)} is incompatible with {tuple(weight.shape)}"
        )

    rank = int(right.shape[0])
    rows = _workspace_rows(
        output_rows=output_rows,
        rank=rank,
        dtype=dtype,
        workspace_mib=workspace_mib,
    )
    prototype_prefix = f"prototypes/{module_type}/{prototype}/left"
    residual_prefix = f"layers/{module_name}/left_residual"
    with torch.no_grad():
        for start in range(0, output_rows, rows):
            stop = min(start + rows, output_rows)
            left = _dequantized_rows(tensors, prototype_prefix, start, stop, dtype=dtype)
            left.add_(_dequantized_rows(tensors, residual_prefix, start, stop, dtype=dtype))
            if left.ndim != 2 or left.shape[1] != rank:
                raise ValueError(
                    f"Reconstructed left-factor shape mismatch at {module_name}: "
                    f"{tuple(left.shape)} is incompatible with rank {rank}"
                )
            weight[start:stop].addmm_(left, right)
            del left
    return weight


def _sun_weight_entries(manifest: Any) -> dict[str, tuple[str, int, str]]:
    result: dict[str, tuple[str, int, str]] = {}
    entries = manifest.metadata.get("module_entries", {})
    for module_type, rows in entries.items():
        for row in rows:
            module_name = str(row["name"])
            result[module_name + ".weight"] = (str(module_type), int(row["prototype"]), module_name)
    return result


def _cache_key(manifest: Any, dtype: torch.dtype) -> str:
    payload = "\n".join(
        [
            str(manifest.base_model),
            str(manifest.target_model),
            str(manifest.tensor_sha256),
            str(dtype),
            "out-of-core-v1",
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def materialize_sun_checkpoint(
    sun_path: str | os.PathLike[str],
    output_root: str | os.PathLike[str],
    *,
    dtype: torch.dtype = torch.float16,
    workspace_mib: int = 256,
    verify_archive: bool = True,
    revision: str | None = None,
    local_files_only: bool = False,
) -> Path:
    """Create an out-of-core reconstructed SafeTensors checkpoint.

    Only one base tensor, one low-rank right factor, and bounded left-factor rows
    are resident while the checkpoint is produced. The result is cached and can
    be loaded repeatedly without reconstructing the SUN delta again.
    """
    manifest = SunArchive.read_manifest(sun_path)
    checkpoint = ShardedCheckpoint.resolve(
        manifest.base_model,
        revision=revision,
        local_files_only=local_files_only,
    )
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / _cache_key(manifest, dtype)
    marker = destination / "yozakura.complete.json"
    index_path = destination / "model.safetensors.index.json"
    if marker.exists() and index_path.exists():
        return destination

    temporary = Path(tempfile.mkdtemp(prefix=destination.name + ".", dir=output_root))
    weight_map: dict[str, str] = {}
    total_size = 0
    sun_entries = _sun_weight_entries(manifest)
    try:
        with SunArchive.open_tensors(sun_path, verify=verify_archive) as (_, sun_tensors):
            names = sorted(checkpoint.weight_map)
            width = max(6, len(str(len(names))))
            for index, name in enumerate(names, start=1):
                value = checkpoint.tensor(name)
                if value.is_floating_point():
                    value = value.to(dtype=dtype)
                entry = sun_entries.get(name)
                if entry is not None:
                    module_type, prototype, module_name = entry
                    value = _patch_projection_weight(
                        value,
                        module_name=module_name,
                        module_type=module_type,
                        prototype=prototype,
                        tensors=sun_tensors,
                        workspace_mib=workspace_mib,
                    )
                value = value.contiguous()
                filename = f"tensor-{index:0{width}d}.safetensors"
                save_file({name: value}, str(temporary / filename))
                weight_map[name] = filename
                total_size += value.numel() * value.element_size()
                del value

        payload = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
        (temporary / "model.safetensors.index.json").write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        (temporary / "yozakura.complete.json").write_text(
            json.dumps(
                {
                    "format": "YOZAKURA-RECONSTRUCTED",
                    "version": 1,
                    "base_model": manifest.base_model,
                    "target_model": manifest.target_model,
                    "sun_sha256": manifest.tensor_sha256,
                    "dtype": str(dtype).removeprefix("torch."),
                    "tensor_count": len(weight_map),
                },
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        if destination.exists():
            shutil.rmtree(destination)
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def load_out_of_core_sun_model(
    sun_path: str,
    *,
    dtype: torch.dtype = torch.float16,
    max_memory: Mapping[int | str, int | str] | None = None,
    offload_folder: str | None = None,
    checkpoint_cache: str = ".yozakura-checkpoints",
    workspace_mib: int = 256,
    verify_archive: bool = True,
    trust_remote_code: bool = False,
    revision: str | None = None,
    local_files_only: bool = False,
):
    """Construct and load a SUN model without full host-RAM residency."""
    try:
        from accelerate import infer_auto_device_map, init_empty_weights, load_checkpoint_and_dispatch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Out-of-core loading requires accelerate>=1.0") from exc

    manifest = SunArchive.read_manifest(sun_path)
    task = str(manifest.metadata.get("task", "causal-lm"))
    adapter, config = resolve_adapter(
        manifest.base_model,
        task,
        trust_remote_code=trust_remote_code,
    )
    reconstructed = materialize_sun_checkpoint(
        sun_path,
        checkpoint_cache,
        dtype=dtype,
        workspace_mib=workspace_mib,
        verify_archive=verify_archive,
        revision=revision,
        local_files_only=local_files_only,
    )

    with init_empty_weights():
        model = adapter.model_class.from_config(config, trust_remote_code=trust_remote_code)
    tie_weights = getattr(model, "tie_weights", None)
    if callable(tie_weights):
        tie_weights()
    model.eval()

    no_split = list(getattr(model, "_no_split_modules", None) or [])
    device_map = infer_auto_device_map(
        model,
        max_memory=dict(max_memory) if max_memory else None,
        no_split_module_classes=no_split or None,
        dtype=dtype,
    )
    offload_dir = offload_folder or ".yozakura-offload"
    Path(offload_dir).mkdir(parents=True, exist_ok=True)
    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=str(reconstructed),
        device_map=device_map,
        offload_folder=offload_dir,
        offload_buffers=True,
        dtype=dtype,
    ).eval()
    frontend = load_frontend(adapter, manifest.target_model, trust_remote_code=trust_remote_code)
    return model, frontend
