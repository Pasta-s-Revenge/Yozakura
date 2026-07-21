# Yozakura 🌸 夜桜

<img width="1983" height="793" alt="Yozakura" src="https://github.com/user-attachments/assets/e5cb7a54-974d-4853-9e74-789e45046294" />

**Run models larger than your machine should normally be able to hold.**

Yozakura is an experimental hypernetwork runtime for making very large language models practical on consumer hardware. Its long-term goal is to let anyone run capable, large-scale models at home by replacing the assumption that every parameter must be stored and loaded as a conventional dense checkpoint.

The core idea is to represent reusable model structure compactly, then reconstruct or generate the required weights with a hypernetwork only when they are needed. Yozakura targets CPU, GPU, and heterogeneous memory systems, with an emphasis on low-capacity machines rather than datacenter-only deployments.

> Yozakura does not yet make arbitrary giant models run instantly on ordinary PCs. The current release is the first storage and reconstruction layer toward that goal.

## Mission

Yozakura aims to reduce three barriers to local LLM inference:

1. **Storage** — avoid distributing another complete copy of mostly shared model weights.
2. **Memory** — reconstruct, stream, cache, and evict model components instead of requiring the full checkpoint to remain resident.
3. **Latency** — develop fused reconstruction and inference paths so compact representations do not create unacceptable runtime overhead.

The intended end state is a local runtime where a consumer machine can execute a model whose conventional checkpoint is substantially larger than its available RAM or VRAM.

## Technical direction

Yozakura treats a model as:

```text
model = shared base structure + generated/reconstructed specialization
```

A compact hypernetwork representation stores the information needed to recover model-specific weights relative to a compatible base model. At runtime, Yozakura will progressively move from eager reconstruction toward layer-wise generation, streaming execution, bounded caches, and hardware-aware scheduling.

The project is guided by measurable constraints rather than nominal parameter counts:

- peak RAM and VRAM usage;
- persistent storage size;
- tokens per second and time to first token;
- reconstruction error and downstream quality;
- energy consumption on consumer hardware.

A representation is useful only when it improves the complete system trade-off. Exact reconstruction that consumes more space than the original is not compression, and a smaller artifact that makes inference impractically slow is not a successful runtime.

## Current implementation: `.sun`

Yozakura currently provides a portable hypernetwork-delta archive named `.sun` (**Shared Universal Network**). A `.sun` file contains:

- a reference to a compatible base model rather than another copy of its weights;
- shared low-rank factor prototypes grouped by module type;
- quantized per-layer residual factors;
- a deterministic manifest with checksums and evaluation metadata;
- non-executable tensor data stored with SafeTensors.

The target and base models must currently share the same architecture and matching linear tensor shapes.

## Why the previous checkpoint approach failed

The previous LayerForge v2 experiment stored one 16-bit dense base per tested layer, then added rank-128 prototype factors and a controller. Its reported `compression_ratio=1.169526` represented a **16.95% expansion**, not compression. Exact reconstruction (`NMSE=0` with unchanged perplexity) did not compensate for the larger representation.

Yozakura therefore evaluates compression at the artifact and runtime-system level, with hard release gates for reconstruction error and effective distribution size.

## Install

```bash
pip install -e .
```

## Build a `.sun` archive

Building a 9B delta can still be memory-intensive even when the resulting artifact is CPU-runnable. Use CPU for compatibility or CUDA for faster construction.

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

The build is rejected and the artifact deleted when the configured reconstruction-error or effective-distribution-size gate is exceeded.

## Run

CPU:

```bash
yozakura run qwythos-9b.sun \
  --device cpu \
  --prompt "Write a Python function that validates a DAG."
```

GPU:

```bash
yozakura run qwythos-9b.sun \
  --device cuda \
  --prompt "Explain speculative decoding."
```

## `.sun` v1 layout

```text
model.sun
├── manifest.json
└── tensors.safetensors
```

`manifest.json` records the format version, base and target model IDs, module mapping, rank, prototype count, evaluation metadata, and SHA-256 checksum of the tensor payload. `tensors.safetensors` contains only non-executable tensor data.

## Roadmap

### Phase 1 — Compact specialization archives

- deterministic `.sun` artifacts;
- quantized low-rank reconstruction;
- strict quality and size gates;
- CPU and CUDA execution through PyTorch and Transformers.

### Phase 2 — Memory-bounded execution

- reconstruct one layer or block at a time;
- bounded RAM/VRAM caches with eviction;
- asynchronous prefetch and reconstruction;
- disk, RAM, and VRAM tiering;
- peak-memory benchmarks on commodity machines.

### Phase 3 — High-speed hypernetwork runtime

- fused reconstruction and matrix-multiplication kernels;
- on-the-fly generated weights without full materialization;
- hardware-aware scheduling across CPU, integrated GPU, and discrete GPU;
- quantized base-model backends, including GGUF-compatible paths where practical.

### Phase 4 — Broad model support

- architecture adapters beyond shape-identical descendants;
- multimodal processor plumbing;
- reusable universal bases and composable specialization modules;
- reproducible quality, speed, memory, storage, and energy benchmarks.

## Current limitations

The current implementation reconstructs deltas once during model loading. It does **not** yet provide:

- execution when the base model itself cannot fit in available memory;
- fused on-the-fly hypernetwork kernels;
- layer-streaming inference;
- GGUF base loading;
- multimodal processor support;
- universal support for unrelated model architectures.

These are central roadmap items, not capabilities of the present release.

## Security and licensing

`.sun` never uses pickle and does not embed Python code. Consumers still need access to the declared base model and must comply with the licenses of both the base and target models.

## Success criteria

Yozakura will consider the direction successful when it can demonstrate, reproducibly, that a consumer machine can run a model substantially larger than its normal resident-memory limit while preserving useful model quality and achieving practical interactive latency.
