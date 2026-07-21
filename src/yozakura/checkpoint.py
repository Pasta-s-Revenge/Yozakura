from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

_INDEX_NAMES = ("model.safetensors.index.json", "pytorch_model.bin.index.json")
_SINGLE_NAMES = ("model.safetensors",)


@dataclass(frozen=True)
class TensorLocation:
    name: str
    shard: Path


class ShardedCheckpoint:
    """Resolve and read one tensor at a time from a SafeTensors checkpoint.

    The reader keeps no model-sized state. Each call opens only the shard that
    contains the requested tensor, allowing callers to construct and evict
    modules incrementally.
    """

    def __init__(self, root: str | Path, weight_map: dict[str, str]):
        self.root = Path(root)
        self.weight_map = dict(weight_map)

    @classmethod
    def resolve(
        cls,
        model: str | Path,
        *,
        revision: str | None = None,
        local_files_only: bool = False,
    ) -> "ShardedCheckpoint":
        candidate = Path(model)
        if candidate.exists():
            root = candidate if candidate.is_dir() else candidate.parent
        else:
            root = Path(
                snapshot_download(
                    repo_id=str(model),
                    revision=revision,
                    allow_patterns=["*.safetensors", "*.safetensors.index.json"],
                    local_files_only=local_files_only,
                )
            )
        return cls.from_directory(root)

    @classmethod
    def from_directory(cls, root: str | Path) -> "ShardedCheckpoint":
        root = Path(root)
        for name in _INDEX_NAMES:
            index_path = root / name
            if index_path.exists():
                payload = json.loads(index_path.read_text(encoding="utf-8"))
                weight_map = payload.get("weight_map")
                if not isinstance(weight_map, dict) or not weight_map:
                    raise ValueError(f"Invalid checkpoint index: {index_path}")
                if any(not str(shard).endswith(".safetensors") for shard in weight_map.values()):
                    raise ValueError("Only SafeTensors checkpoints are supported")
                return cls(root, {str(k): str(v) for k, v in weight_map.items()})

        for name in _SINGLE_NAMES:
            shard = root / name
            if shard.exists():
                with safe_open(str(shard), framework="pt", device="cpu") as reader:
                    return cls(root, {key: name for key in reader.keys()})
        raise FileNotFoundError(f"No SafeTensors checkpoint found in {root}")

    def location(self, name: str) -> TensorLocation:
        try:
            shard_name = self.weight_map[name]
        except KeyError as exc:
            raise KeyError(f"Tensor not found in checkpoint: {name}") from exc
        shard = self.root / shard_name
        if not shard.exists():
            raise FileNotFoundError(f"Missing checkpoint shard: {shard}")
        return TensorLocation(name=name, shard=shard)

    def tensor(
        self,
        name: str,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        location = self.location(name)
        with safe_open(str(location.shard), framework="pt", device="cpu") as reader:
            value = reader.get_tensor(name)
        if dtype is not None or str(device) != "cpu":
            value = value.to(device=device, dtype=dtype or value.dtype)
        return value

    def tensors(self, names: Iterator[str]) -> Iterator[tuple[str, torch.Tensor]]:
        for name in names:
            yield name, self.tensor(name)
