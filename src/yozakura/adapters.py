from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import torch.nn as nn
import transformers
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForImageClassification,
    AutoModelForMaskedLM,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    AutoTokenizer,
)

try:
    from transformers.pytorch_utils import Conv1D
except ImportError:  # pragma: no cover - compatibility fallback
    Conv1D = ()  # type: ignore[assignment,misc]


def _first_transformers_class(*names: str, fallback: type) -> type:
    """Resolve renamed optional AutoModel classes without import-time failure."""
    for name in names:
        candidate = getattr(transformers, name, None)
        if candidate is not None:
            return candidate
    return fallback


AutoModelForMultimodal = _first_transformers_class(
    "AutoModelForImageTextToText",
    "AutoModelForMultimodalLM",
    "AutoModelForVision2Seq",
    fallback=AutoModel,
)


@dataclass(frozen=True, slots=True)
class HFAdapter:
    task: str
    model_class: type
    frontend_class: type | None
    generative: bool


ADAPTERS: dict[str, HFAdapter] = {
    "causal-lm": HFAdapter("causal-lm", AutoModelForCausalLM, AutoTokenizer, True),
    "seq2seq-lm": HFAdapter("seq2seq-lm", AutoModelForSeq2SeqLM, AutoTokenizer, True),
    "masked-lm": HFAdapter("masked-lm", AutoModelForMaskedLM, AutoTokenizer, False),
    "sequence-classification": HFAdapter("sequence-classification", AutoModelForSequenceClassification, AutoTokenizer, False),
    "image-classification": HFAdapter("image-classification", AutoModelForImageClassification, AutoProcessor, False),
    "vision2seq": HFAdapter("vision2seq", AutoModelForMultimodal, AutoProcessor, True),
    "speech-seq2seq": HFAdapter("speech-seq2seq", AutoModelForSpeechSeq2Seq, AutoProcessor, True),
    "base": HFAdapter("base", AutoModel, AutoProcessor, False),
}

FAMILY_TASK_HINTS: dict[str, str] = {
    "llama": "causal-lm", "mistral": "causal-lm", "mixtral": "causal-lm",
    "qwen2": "causal-lm", "qwen3": "causal-lm", "gemma": "causal-lm",
    "gemma2": "causal-lm", "phi": "causal-lm", "phi3": "causal-lm",
    "gpt2": "causal-lm", "gpt_neox": "causal-lm", "falcon": "causal-lm",
    "bloom": "causal-lm", "mpt": "causal-lm", "opt": "causal-lm",
    "t5": "seq2seq-lm", "mt5": "seq2seq-lm", "bart": "seq2seq-lm",
    "pegasus": "seq2seq-lm", "bert": "masked-lm", "roberta": "masked-lm",
    "deberta": "masked-lm", "deberta-v2": "masked-lm", "electra": "masked-lm",
    "vit": "image-classification", "swin": "image-classification",
    "convnext": "image-classification", "whisper": "speech-seq2seq",
    "vision-encoder-decoder": "vision2seq", "llava": "vision2seq",
    "idefics": "vision2seq", "clip": "base", "siglip": "base",
    "qwen3_5": "vision2seq", "qwen3_5_moe": "vision2seq",
    "qwen2_vl": "vision2seq", "qwen2_5_vl": "vision2seq", "qwen3_vl": "vision2seq",
}


def resolve_adapter(model_id: str, task: str = "auto", *, trust_remote_code: bool = False) -> tuple[HFAdapter, Any]:
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    if task != "auto":
        if task not in ADAPTERS:
            raise ValueError(f"Unknown task {task!r}; choose from {sorted(ADAPTERS)}")
        return ADAPTERS[task], config
    model_type = str(getattr(config, "model_type", ""))
    hinted = FAMILY_TASK_HINTS.get(model_type)
    if hinted:
        return ADAPTERS[hinted], config
    if getattr(config, "is_encoder_decoder", False):
        return ADAPTERS["seq2seq-lm"], config
    architectures = " ".join(getattr(config, "architectures", None) or [])
    for needle, candidate in (
        ("ForCausalLM", "causal-lm"),
        ("ForImageTextToText", "vision2seq"),
        ("ForVision2Seq", "vision2seq"),
        ("ForMultimodal", "vision2seq"),
        ("ForConditionalGeneration", "vision2seq" if hasattr(config, "vision_config") else "seq2seq-lm"),
        ("ForMaskedLM", "masked-lm"),
        ("ForSequenceClassification", "sequence-classification"),
        ("ForImageClassification", "image-classification"),
        ("ForSpeechSeq2Seq", "speech-seq2seq"),
    ):
        if needle in architectures:
            return ADAPTERS[candidate], config
    return ADAPTERS["base"], config


def _is_projection(module: nn.Module) -> bool:
    """Return true for standard Linear and GPT-style Conv1D projections."""
    return isinstance(module, nn.Linear) or (Conv1D and isinstance(module, Conv1D))


def matching_linears(
    base: nn.Module,
    target: nn.Module,
    modules: tuple[str, ...] | None = None,
) -> tuple[dict[str, nn.Module], dict[str, nn.Module], tuple[str, ...]]:
    base_all = {n: m for n, m in base.named_modules() if _is_projection(m)}
    target_all = {n: m for n, m in target.named_modules() if _is_projection(m)}
    common = {
        n for n in base_all.keys() & target_all.keys()
        if base_all[n].weight.ndim == 2
        and base_all[n].weight.shape == target_all[n].weight.shape
    }

    if modules == ("*",):
        selected = tuple(sorted({n.rsplit(".", 1)[-1] for n in common}))
    elif modules:
        common = {n for n in common if n.rsplit(".", 1)[-1] in modules}
        selected = modules
    else:
        counts = Counter(n.rsplit(".", 1)[-1] for n in common)
        selected = tuple(sorted(s for s, count in counts.items() if count >= 2))
        common = {n for n in common if n.rsplit(".", 1)[-1] in selected}

    return ({n: base_all[n] for n in common}, {n: target_all[n] for n in common}, selected)


def load_frontend(adapter: HFAdapter, model_id: str, *, trust_remote_code: bool = False):
    if adapter.frontend_class is None:
        return None
    return adapter.frontend_class.from_pretrained(model_id, trust_remote_code=trust_remote_code)
