from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from yozakura.cli import _parse_max_memory
from yozakura.runtime import model_input_device


def test_parse_max_memory_supports_gpu_cpu_and_disk() -> None:
    assert _parse_max_memory(["0=8GiB", "cpu=24GiB", "disk=100GiB"]) == {
        0: "8GiB",
        "cpu": "24GiB",
        "disk": "100GiB",
    }


@pytest.mark.parametrize("value", ["cpu", "=8GiB", "cpu="])
def test_parse_max_memory_rejects_invalid_entries(value: str) -> None:
    with pytest.raises(SystemExit, match="DEVICE=LIMIT"):
        _parse_max_memory([value])


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(8, 4)
        self.proj = nn.Linear(4, 4)

    def get_input_embeddings(self) -> nn.Module:
        return self.embed


def test_model_input_device_prefers_embeddings() -> None:
    model = TinyModel()
    assert model_input_device(model) == torch.device("cpu")
