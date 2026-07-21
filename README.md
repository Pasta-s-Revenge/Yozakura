# Yozakura 🌸 夜桜

<img width="1983" height="793" alt="Yozakura" src="https://github.com/user-attachments/assets/e5cb7a54-974d-4853-9e74-789e45046294" />

**Run models larger than your machine should normally be able to hold.**

Yozakura is an experimental hypernetwork runtime for making very large language models practical on consumer hardware. Its long-term goal is to let anyone run capable, large-scale models at home by replacing the assumption that every parameter must be stored and loaded as a conventional dense checkpoint.

The core idea is to represent reusable model structure compactly, then reconstruct or generate the required weights with a hypernetwork only when they are needed. Yozakura targets CPU, GPU, and heterogeneous memory systems, with an emphasis on low-capacity machines rather than datacenter-only deployments.

> The current runtime supports bounded SUN reconstruction, GPU/CPU/disk tiering, and an end-to-end out-of-core construction path for compatible SafeTensors checkpoints. It does not yet provide fused on-the-fly weight generation.

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

A compact hypernetwork representation stores the information needed to recover model-specific weights relative to a compatible base model. The runtime can reconstruct a SafeTensors checkpoint one tensor at a time, instantiate the model on the meta device, and load weights directly into GPU, CPU, or disk tiers.

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

## Colab demo

Open [`notebooks/Yozakura_Colab_Demo.ipynb`](notebooks/Yozakura_Colab_Demo.ipynb) in Google Colab to run a tiny end-to-end smoke test and an out-of-core `.sun` template. For this private repository, add a Colab Secret named `GITHUB_TOKEN` with read access to the repository.

The notebook demonstrates:

- creation of a tiny `.sun` smoke artifact;
- one-tensor-at-a-time base checkpoint reconstruction;
- meta-device initialization;
- GPU/CPU/disk placement;
- generation and reconstructed-checkpoint cache reuse;
- upload and execution of a user-provided `.sun` file.

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

## Runtime modes

### Eager CPU or GPU

The complete base model is loaded, then SUN deltas are applied with bounded reconstruction workspace.

```bash
yozakura run model.sun \
  --device cuda \
  --workspace-mib 256 \
  --prompt "Explain speculative decoding."
```

### Tiered GPU, CPU, and disk

The complete model is reconstructed in CPU RAM first, then Accelerate places modules across memory tiers.

```bash
yozakura run model.sun \
  --device auto \
  --max-memory 0=8GiB \
  --max-memory cpu=24GiB \
  --offload-folder .yozakura-offload \
  --prompt "Explain bounded-memory inference."
```

### Fully out-of-core construction and execution

This mode does not materialize the complete base checkpoint in host RAM during construction.

```bash
yozakura run model.sun \
  --device out-of-core \
  --dtype float16 \
  --max-memory 0=8GiB \
  --max-memory cpu=16GiB \
  --offload-folder .yozakura-offload \
  --checkpoint-cache .yozakura-checkpoints \
  --workspace-mib 256 \
  --prompt "Explain out-of-core inference."
```

The first run performs this pipeline:

```text
base SafeTensors shard
        ↓ one tensor
bounded SUN reconstruction
        ↓
reconstructed SafeTensors cache
        ↓
meta-initialized Transformers model
        ↓
GPU / CPU / disk device map
        ↓
generation through Accelerate hooks
```

Subsequent runs reuse the reconstructed checkpoint when the SUN checksum, base model identity, dtype, and format version match.

## `.sun` v1 layout

```text
model.sun
├── manifest.json
└── tensors.safetensors
```

`manifest.json` records the format version, base and target model IDs, module mapping, rank, prototype count, evaluation metadata, and SHA-256 checksum of the tensor payload. `tensors.safetensors` contains only non-executable tensor data.

## Memory guarantees

During out-of-core cache construction, model-weight residency is approximately bounded by:

```text
current base tensor
+ current low-rank right factor
+ bounded left-factor row chunks
+ SafeTensors serialization overhead
```

The complete model structure is created on `meta`; reconstructed weights are then loaded directly into the inferred memory tiers.

## Current trade-offs and limitations

- the first out-of-core run requires persistent storage approximately equal to the reconstructed model size;
- the initial bounded writer stores one tensor per SafeTensors file, which increases filesystem overhead;
- only SafeTensors base checkpoints are accepted by the out-of-core reader;
- base and target models must have compatible architecture and matching selected tensor shapes;
- reconstruction is performed before inference rather than fused into matrix multiplication;
- GGUF base loading is not implemented;
- multimodal processor support is incomplete;
- unrelated model architectures cannot yet share one universal base.

## Roadmap

### Phase 1 — Compact specialization archives

- deterministic `.sun` artifacts;
- quantized low-rank reconstruction;
- strict quality and size gates;
- CPU and CUDA execution through PyTorch and Transformers.

### Phase 2 — Memory-bounded execution

- bounded reconstruction workspace;
- GPU, CPU, and disk tiering;
- SafeTensors tensor-level checkpoint reading;
- meta-device out-of-core model construction;
- persistent reconstructed-checkpoint caching;
- peak-memory benchmarks on commodity machines.

### Phase 3 — High-speed hypernetwork runtime

- direct reconstruction into offload storage;
- asynchronous shard and layer prefetch;
- fused reconstruction and matrix-multiplication kernels;
- on-the-fly generated weights without a full reconstructed checkpoint;
- hardware-aware scheduling across CPU, integrated GPU, and discrete GPU;
- quantized base-model backends, including GGUF-compatible paths where practical.

### Phase 4 — Broad model support

- architecture adapters beyond shape-identical descendants;
- multimodal processor plumbing;
- reusable universal bases and composable specialization modules;
- reproducible quality, speed, memory, storage, and energy benchmarks.

## Security and licensing

`.sun` never uses pickle and does not embed Python code. Consumers still need access to the declared base model and must comply with the licenses of both the base and target models.

## Success criteria

Yozakura will consider the direction successful when it can demonstrate, reproducibly, that a consumer machine can run a model substantially larger than its normal resident-memory limit while preserving useful model quality and achieving practical interactive latency.
