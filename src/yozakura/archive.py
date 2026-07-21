from __future__ import annotations

import hashlib
import json
import os
import tempfile
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
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
            tensor_bytes = tensor_path.read_bytes()
            manifest.tensor_sha256 = hashlib.sha256(tensor_bytes).hexdigest()
            manifest_bytes = json.dumps(asdict(manifest), ensure_ascii=False, sort_keys=True, indent=2).encode()

            tmp = out.with_suffix(out.suffix + ".tmp")
            with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED) as zf:
                for name, payload in (("manifest.json", manifest_bytes), ("tensors.safetensors", tensor_bytes)):
                    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                    info.compress_type = zipfile.ZIP_STORED
                    info.external_attr = 0o644 << 16
                    zf.writestr(info, payload)
            os.replace(tmp, out)
        return out

    @staticmethod
    def read(path: str | os.PathLike[str], device: str = "cpu") -> tuple[SunManifest, dict[str, torch.Tensor]]:
        src = Path(path)
        with zipfile.ZipFile(src, "r") as zf:
            names = set(zf.namelist())
            if names != {"manifest.json", "tensors.safetensors"}:
                raise ValueError(f"Invalid .sun members: {sorted(names)}")
            raw_manifest = json.loads(zf.read("manifest.json"))
            tensor_bytes = zf.read("tensors.safetensors")

        manifest = SunManifest(**raw_manifest)
        manifest.validate()
        digest = hashlib.sha256(tensor_bytes).hexdigest()
        if digest != manifest.tensor_sha256:
            raise ValueError(".sun tensor checksum mismatch")

        with tempfile.NamedTemporaryFile(suffix=".safetensors") as f:
            f.write(tensor_bytes)
            f.flush()
            tensors = load_file(f.name, device=device)
        return manifest, tensors
