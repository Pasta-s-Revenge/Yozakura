import torch.nn as nn

from yozakura.adapters import ADAPTERS, FAMILY_TASK_HINTS, matching_linears


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleDict({"q_proj": nn.Linear(4, 4), "up_proj": nn.Linear(4, 8)}),
            nn.ModuleDict({"q_proj": nn.Linear(4, 4), "up_proj": nn.Linear(4, 8)}),
        ])
        self.lm_head = nn.Linear(4, 10)


def test_auto_module_discovery_omits_one_off_head():
    base, target = TinyModel(), TinyModel()
    base_linears, target_linears, suffixes = matching_linears(base, target)
    assert suffixes == ("q_proj", "up_proj")
    assert set(base_linears) == set(target_linears)
    assert all(not name.endswith("lm_head") for name in base_linears)


def test_explicit_module_filter():
    base, target = TinyModel(), TinyModel()
    base_linears, _, suffixes = matching_linears(base, target, ("q_proj",))
    assert suffixes == ("q_proj",)
    assert len(base_linears) == 2


def test_registry_has_major_modalities():
    assert {"causal-lm", "seq2seq-lm", "image-classification", "vision2seq", "speech-seq2seq"} <= ADAPTERS.keys()
    assert FAMILY_TASK_HINTS["llama"] == "causal-lm"
    assert FAMILY_TASK_HINTS["whisper"] == "speech-seq2seq"
