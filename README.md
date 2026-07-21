# Yozakura　🌸　夜桜　
<img width="1983" height="793" alt="image" src="https://github.com/user-attachments/assets/e5cb7a54-974d-4853-9e74-789e45046294" />

Yozakura is a portable hypernetwork framework for CPU/GPU Transformers inference. It introduces `.sun` (Shared Universal Network), a deterministic archive containing a quantized low-rank hypernetwork delta and a reference to its base model.

## Why the previous checkpoint failed

The LayerForge v2 configuration used one 16-bit dense base per tested layer and then added rank-128 prototype factors plus a controller. Its `compression_ratio=1.169526` therefore meant a 16.95% expansion. Exact reconstruction (`NMSE=0`, unchanged perplexity) is not a compression result.

Yozakura instead stores:

- a base model reference, not another copy of the base weights;
- shared low-rank factor prototypes per module type;
- int8 per-layer factor residuals;
- a manifest with checksums and a hard release gate.

The target and base must have the same architecture and matching linear tensor shapes. For Qwythos, use its Qwen3.5-9B ancestor as the base.

## Install

```bash
pip install -e .
```

## Build Qwythos `.sun`

Building a 9B delta is memory-intensive even when the resulting artifact is CPU-runnable. Use CPU for maximum compatibility or CUDA to build faster.

```bash
yozakura build \
  --base Qwen/Qwen3.5-9B \
  --target empero-ai/Qwythos-9B-Claude-Mythos-5-1M \
  --output qwythos-9b.sun \
  --modules gate_proj,up_proj,down_proj \
  --rank 32 \
  --prototypes 4 \
  --device cpu
```

The build is rejected and the artifact deleted when either the low-rank delta NMSE or the effective distribution ratio exceeds the configured release gate.

## Run on CPU

```bash
yozakura run qwythos-9b.sun \
  --device cpu \
  --prompt "Write a Python function that validates a DAG."
```

## Run on GPU

```bash
yozakura run qwythos-9b.sun --device cuda --prompt "Explain speculative decoding."
```

## `.sun` v1 layout

```text
model.sun
├── manifest.json
└── tensors.safetensors
```

`manifest.json` records format/version, base and target model IDs, module mapping, rank, prototype count, evaluation metadata, and SHA-256 of the tensor payload. `tensors.safetensors` stores only non-executable tensor data.

## Security and distribution

`.sun` never uses pickle and does not embed Python code. Consumers still need access to the declared base model and must comply with both base and target model licenses. Qwythos is published under Apache-2.0 according to its model card.

## Current scope

This first implementation applies reconstructed deltas once during model loading. It supports CPU and CUDA through PyTorch/Transformers. It does not yet provide fused on-the-fly kernels, GGUF base loading, multimodal processor plumbing, or streaming reconstruction for machines that cannot hold the base model plus one layer workspace.
