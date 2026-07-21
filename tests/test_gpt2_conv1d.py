import torch
import torch.nn as nn
from transformers.pytorch_utils import Conv1D

from yozakura.adapters import matching_linears
from yozakura.cli import _parse_modules


class TinyGPT2Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_attn = Conv1D(12, 4)
        self.c_proj = Conv1D(4, 4)


class TinyGPT2(nn.Module):
    def __init__(self):
        super().__init__()
        self.h = nn.ModuleList([TinyGPT2Block(), TinyGPT2Block()])
        self.lm_head = nn.Linear(4, 4, bias=False)


def test_modules_all_is_wildcard():
    assert _parse_modules("all") == ("*",)
    assert _parse_modules("*") == ("*",)
    assert _parse_modules("auto") is None


def test_matching_linears_includes_gpt2_conv1d():
    base = TinyGPT2()
    target = TinyGPT2()
    base_modules, target_modules, selected = matching_linears(base, target, ("*",))

    assert "h.0.c_attn" in base_modules
    assert "h.1.c_proj" in target_modules
    assert "c_attn" in selected
    assert all(module.weight.ndim == 2 for module in base_modules.values())


def test_auto_keeps_repeated_conv1d_and_omits_one_off_head():
    base = TinyGPT2()
    target = TinyGPT2()
    base_modules, _, selected = matching_linears(base, target, None)

    assert set(selected) == {"c_attn", "c_proj"}
    assert "lm_head" not in selected
    assert len(base_modules) == 4
