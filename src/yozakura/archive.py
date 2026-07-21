from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

SUN_MAGIC = "YOZAKURA-SUN"
SUN_VERSION = 1


@dataclass(slots=True)
class SunManifest:
    base_model: str
    target_model: str
    modules: list[str]
    rank: int
    prototypes_per_module: int
    quantization: str = "int8-symmetric"
    format: str = SUN_MAGIC
    format_version: int = SUN_VERSION
    architecture: str = "shared-low-rank-prototype-hypernetwork"
    metadata: dict[str, Any] = field(default_factory=dict)
    tensor_sha256: str = ""

    def validate(self) -> None:
        if self.format != SUN_MAGIC or self.format_version != SUN_VERSION:
            raise ValueError("Unsupported .sun format")
        if not self.base_model or not self.target_model:
            raise ValueError("base_model and target_model are required")
        if self.rank <= 0 or self.prototypes_per_module <= 0:
            raise ValueError("rank and prototypes_per_module must be positive")
        if self.quantization not in {"int8-symmetric", "fp16"}:
            raise ValueError(f"Unsupported quantization: {self.quantization}")


def _sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


class SunArchive:
    """Read/write deterministic .sun ZIP archives containing only tensor deltas."""

    @staticmethod
    def write(path: str | os.PathLike[str], manifest: SunManifest, tensors: dict[str, torch.Tensor]) -> Path:
        manifest.validate()
        out = Path(path)
        if out.suffix != ".sun":
            out = out.with_suffix(".sun")
        out.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory() as td:
            tensor_path = Path(td) / "tensors.safetensors"
            contiguous = {k: v.detach().cpu().contiguous() for k, v in tensors.items()}
            save_file(contiguous, str(tensor_path))
            del contiguous
            manifest.tensor_sha256 = _sha256_file(tensor_path)
            manifest_bytes = json.dumps(asdict(manifest), ensure_ascii=False, sort_keys=True, indent=2).encode()

            tmp = out.with_suffix(out.suffix + ".tmp")
            with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED) as zf:
                info = zipfile.ZipInfo("manifest.json", date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_STORED
                info.external_attr = 0o644 << 16
                zf.writestr(info, manifest_bytes)

                info = zipfile.ZipInfo("tensors.safetensors", date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_STORED
                info.external_attr = 0o644 << 16
                with tensor_path.open("rb") as src, zf.open(info, "w") as dst:
                    shutil.copyfileobj(src, dst, length=8 << 20)
            os.replace(tmp, out)
        return out

    @staticmethod
    def read_manifest(path: str | os.PathLike[str]) -> SunManifest:
        with zipfile.ZipFile(Path(path), "r") as zf:
            names = set(zf.namelist())
            if names != {"manifest.json", "tensors.safetensors"}:
                raise ValueError(f"Invalid .sun members: {sorted(names)}")
            raw_manifest = json.loads(zf.read("manifest.json"))
        manifest = SunManifest(**raw_manifest)
        manifest.validate()
        return manifest

    @staticmethod
    @contextmanager
    def open_tensors(
        path: str | os.PathLike[str],
        *,
        device: str = "cpu",
        verify: bool = True,
    ) -> Iterator[tuple[SunManifest, Any]]:
        src = Path(path)
        manifest = SunArchive.read_manifest(src)
        with tempfile.TemporaryDirectory() as td:
            tensor_path = Path(td) / "tensors.safetensors"
            with zipfile.ZipFile(src, "r") as zf, zf.open("tensors.safetensors", "r") as packed, tensor_path.open("wb") as unpacked:
                shutil.copyfileobj(packed, unpacked, length=8 << 20)
            if verify and _sha256_file(tensor_path) != manifest.tensor_sha256:
                raise ValueError(".sun tensor checksum mismatch")
            with safe_open(str(tensor_path), framework="pt", device=device) as tensors:
                yield manifest, tensors

    @staticmethod
    def read(path: str | os.PathLike[str], device: str = "cpu") -> tuple[SunManifest, dict[str, torch.Tensor]]:
        """Compatibility API. Prefer read_manifest/open_tensors for bounded peak RSS."""
        with SunArchive.open_tensors(path, device=device) as (manifest, reader):
            tensors = {key: reader.get_tensor(key) for key in reader.keys()}
        return manifest, tensors
