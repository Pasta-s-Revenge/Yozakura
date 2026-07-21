from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from yozakura.archive import SunArchive, SunManifest
from yozakura.codec import quantize_symmetric
from yozakura.out_of_core import materialize_sun_checkpoint


def _store_quantized(tensors: dict[str, torch.Tensor], prefix: str, value: torch.Tensor) -> None:
    q, scale = quantize_symmetric(value)
    tensors[prefix + ".q"] = q
    tensors[prefix + ".scale"] = scale


def _build_fixture(tmp_path: Path) -> tuple[Path, torch.Tensor, torch.Tensor]:
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    base_weight = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    untouched = torch.tensor([1, 2, 3], dtype=torch.int64)
    save_file(
        {"layer.weight": base_weight, "counter": untouched},
        str(base_dir / "model.safetensors"),
    )

    left = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    right = torch.tensor([[1.0, 2.0, 3.0, 4.0], [-1.0, 0.0, 1.0, 2.0]])
    zeros_left = torch.zeros_like(left)
    zeros_right = torch.zeros_like(right)
    tensors: dict[str, torch.Tensor] = {}
    _store_quantized(tensors, "prototypes/proj/0/left", left)
    _store_quantized(tensors, "prototypes/proj/0/right", right)
    _store_quantized(tensors, "layers/layer/left_residual", zeros_left)
    _store_quantized(tensors, "layers/layer/right_residual", zeros_right)

    manifest = SunManifest(
        base_model=str(base_dir),
        target_model=str(base_dir),
        modules=["proj"],
        rank=2,
        prototypes_per_module=1,
        metadata={
            "task": "causal-lm",
            "module_entries": {"proj": [{"name": "layer", "prototype": 0}]},
        },
    )
    archive = SunArchive.write(tmp_path / "model.sun", manifest, tensors)
    return archive, base_weight + left @ right, untouched


def _read_checkpoint_tensor(checkpoint: Path, name: str) -> torch.Tensor:
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())
    shard = checkpoint / index["weight_map"][name]
    with safe_open(str(shard), framework="pt", device="cpu") as reader:
        return reader.get_tensor(name)


def test_materialize_reconstructs_delta_without_full_model(tmp_path: Path) -> None:
    archive, expected, untouched = _build_fixture(tmp_path)

    checkpoint = materialize_sun_checkpoint(
        archive,
        tmp_path / "cache",
        dtype=torch.float32,
        workspace_mib=1,
    )

    actual = _read_checkpoint_tensor(checkpoint, "layer.weight")
    actual_untouched = _read_checkpoint_tensor(checkpoint, "counter")
    assert torch.allclose(actual, expected, atol=0.05)
    assert torch.equal(actual_untouched, untouched)


def test_materialize_reuses_complete_cache(tmp_path: Path) -> None:
    archive, _, _ = _build_fixture(tmp_path)
    cache = tmp_path / "cache"

    first = materialize_sun_checkpoint(archive, cache, dtype=torch.float32)
    marker = first / "yozakura.complete.json"
    original = marker.stat().st_mtime_ns
    second = materialize_sun_checkpoint(archive, cache, dtype=torch.float32)

    assert second == first
    assert marker.stat().st_mtime_ns == original


def test_materialize_rejects_invalid_workspace(tmp_path: Path) -> None:
    archive, _, _ = _build_fixture(tmp_path)

    try:
        materialize_sun_checkpoint(archive, tmp_path / "cache", workspace_mib=0)
    except ValueError as exc:
        assert "workspace_mib must be positive" in str(exc)
    else:
        raise AssertionError("expected invalid workspace to fail")
