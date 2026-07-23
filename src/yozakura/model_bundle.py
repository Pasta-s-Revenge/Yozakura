from __future__ import annotations

import json
import os
import shutil
import zipfile
from dataclasses import asdict
from pathlib import Path

from huggingface_hub import snapshot_download

from .archive import SunArchive

BUNDLE_FORMAT = "YOZAKURA-MODEL-BUNDLE"
BUNDLE_VERSION = 1
BUNDLE_MANIFEST = "bundle.json"
BUNDLE_ARCHIVE = "model.sun"

_FRONTEND_PATTERNS = [
    "*.json",
    "*.txt",
    "*.model",
    "*.tiktoken",
    "*.py",
    "tokenizer*",
    "processor*",
    "preprocessor*",
    "merges.txt",
    "vocab.*",
    "chat_template*",
]


def build_model_bundle(
    archive: str | os.PathLike[str],
    output: str | os.PathLike[str],
    *,
    revision: str | None = None,
    force: bool = False,
) -> Path:
    """Create an offline model directory containing base weights, frontend, and SUN delta."""
    archive_path = Path(archive).resolve()
    manifest = SunArchive.read_manifest(archive_path)
    destination = Path(output).resolve()
    if destination.exists():
        if not force:
            raise FileExistsError(f"Bundle destination already exists: {destination}")
        shutil.rmtree(destination)
    destination.mkdir(parents=True)

    base_dir = destination / "base"
    frontend_dir = destination / "frontend"
    try:
        snapshot_download(
            repo_id=manifest.base_model,
            revision=revision,
            local_dir=base_dir,
        )
        snapshot_download(
            repo_id=manifest.target_model,
            revision=revision,
            local_dir=frontend_dir,
            allow_patterns=_FRONTEND_PATTERNS,
        )
        shutil.copy2(archive_path, destination / BUNDLE_ARCHIVE)
        payload = {
            "format": BUNDLE_FORMAT,
            "format_version": BUNDLE_VERSION,
            "archive": BUNDLE_ARCHIVE,
            "base": "base",
            "frontend": "frontend",
            "base_model": manifest.base_model,
            "target_model": manifest.target_model,
            "revision": revision,
        }
        (destination / BUNDLE_MANIFEST).write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
    except BaseException:
        shutil.rmtree(destination, ignore_errors=True)
        raise
    return destination


def resolve_model_source(source: str | os.PathLike[str]) -> Path:
    """Resolve a .sun file or self-contained bundle directory to a runnable .sun file."""
    path = Path(source).resolve()
    if path.is_file():
        return path
    bundle_path = path / BUNDLE_MANIFEST
    if not path.is_dir() or not bundle_path.is_file():
        raise FileNotFoundError(f"Expected a .sun file or Yozakura model bundle: {path}")

    payload = json.loads(bundle_path.read_text(encoding="utf-8"))
    if payload.get("format") != BUNDLE_FORMAT or payload.get("format_version") != BUNDLE_VERSION:
        raise ValueError("Unsupported Yozakura model bundle")

    archive_path = (path / payload["archive"]).resolve()
    base_path = (path / payload["base"]).resolve()
    frontend_path = (path / payload["frontend"]).resolve()
    if not archive_path.is_file() or not base_path.is_dir() or not frontend_path.is_dir():
        raise ValueError("Incomplete Yozakura model bundle")

    manifest = SunArchive.read_manifest(archive_path)
    manifest.base_model = str(base_path)
    manifest.target_model = str(frontend_path)
    resolved = path / ".yozakura-resolved.sun"
    temporary = resolved.with_suffix(".sun.tmp")
    with zipfile.ZipFile(archive_path, "r") as source_zip, zipfile.ZipFile(
        temporary, "w", compression=zipfile.ZIP_STORED
    ) as target_zip:
        info = zipfile.ZipInfo("manifest.json", date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_STORED
        info.external_attr = 0o644 << 16
        target_zip.writestr(
            info,
            json.dumps(asdict(manifest), ensure_ascii=False, sort_keys=True, indent=2).encode(),
        )
        info = zipfile.ZipInfo("tensors.safetensors", date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_STORED
        info.external_attr = 0o644 << 16
        with source_zip.open("tensors.safetensors", "r") as src, target_zip.open(info, "w") as dst:
            shutil.copyfileobj(src, dst, length=8 << 20)
    os.replace(temporary, resolved)
    return resolved
