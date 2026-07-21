import zipfile

import pytest
import torch

from yozakura.archive import SunArchive, SunManifest


def test_sun_roundtrip(tmp_path):
    path = tmp_path / "tiny.sun"
    manifest = SunManifest(base_model="base", target_model="target", modules=["gate_proj"], rank=2, prototypes_per_module=1)
    tensors = {"x": torch.tensor([[1, 2], [3, 4]], dtype=torch.int8)}
    SunArchive.write(path, manifest, tensors)
    loaded, values = SunArchive.read(path)
    assert loaded.base_model == "base"
    assert torch.equal(values["x"], tensors["x"])


def test_checksum_rejects_tampering(tmp_path):
    path = tmp_path / "bad.sun"
    manifest = SunManifest(base_model="base", target_model="target", modules=["x"], rank=1, prototypes_per_module=1)
    SunArchive.write(path, manifest, {"x": torch.ones(1)})
    with zipfile.ZipFile(path, "a") as zf:
        zf.writestr("tensors.safetensors", b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        SunArchive.read(path)
