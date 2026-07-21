from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import save_file

from yozakura.checkpoint import ShardedCheckpoint


def test_reads_tensor_from_indexed_shard(tmp_path) -> None:
    save_file({"model.layers.0.weight": torch.arange(6).reshape(2, 3)}, str(tmp_path / "model-00001-of-00002.safetensors"))
    save_file({"model.layers.1.weight": torch.ones(3, 2)}, str(tmp_path / "model-00002-of-00002.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 48},
                "weight_map": {
                    "model.layers.0.weight": "model-00001-of-00002.safetensors",
                    "model.layers.1.weight": "model-00002-of-00002.safetensors",
                },
            }
        ),
        encoding="utf-8",
    )

    checkpoint = ShardedCheckpoint.from_directory(tmp_path)

    assert checkpoint.location("model.layers.1.weight").shard.name == "model-00002-of-00002.safetensors"
    assert torch.equal(checkpoint.tensor("model.layers.0.weight"), torch.arange(6).reshape(2, 3))


def test_reads_single_file_checkpoint(tmp_path) -> None:
    save_file({"weight": torch.tensor([1.0, 2.0])}, str(tmp_path / "model.safetensors"))

    checkpoint = ShardedCheckpoint.from_directory(tmp_path)

    assert torch.equal(checkpoint.tensor("weight", dtype=torch.float16), torch.tensor([1.0, 2.0], dtype=torch.float16))


def test_rejects_pickle_checkpoint_index(tmp_path) -> None:
    (tmp_path / "pytorch_model.bin.index.json").write_text(
        json.dumps({"weight_map": {"weight": "pytorch_model-00001-of-00001.bin"}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Only SafeTensors"):
        ShardedCheckpoint.from_directory(tmp_path)


def test_missing_tensor_has_clear_error(tmp_path) -> None:
    save_file({"weight": torch.ones(1)}, str(tmp_path / "model.safetensors"))
    checkpoint = ShardedCheckpoint.from_directory(tmp_path)

    with pytest.raises(KeyError, match="Tensor not found"):
        checkpoint.tensor("missing")
