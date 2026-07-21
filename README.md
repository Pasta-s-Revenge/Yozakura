# Yozakura 🌸 夜桜

<img width="1983" height="793" alt="Yozakura" src="https://github.com/user-attachments/assets/e5cb7a54-974d-4853-9e74-789e45046294" />

**Run models larger than your machine should normally be able to hold.**

Yozakura is an experimental hypernetwork runtime for making very large language models practical on consumer hardware. Its goal is to let anyone run capable, large-scale models at home by replacing the assumption that every parameter must be stored and loaded as a conventional dense checkpoint.

The core idea is to represent reusable model structure compactly, reconstruct model-specific weights from a `.sun` hypernetwork archive, and place those weights across GPU, CPU RAM, and disk without requiring the complete model to reside in one memory tier.

## Mission

Yozakura targets three barriers to local LLM inference:

1. **Storage** — avoid distributing another complete copy of mostly shared model weights.
2. **Memory** — reconstruct, stream, cache, and evict model components instead of requiring the full checkpoint to remain resident.
3. **Latency** — develop fused reconstruction and inference paths so compact representations do not create unacceptable runtime overhead.

## Technical model

```text
model = shared base checkpoint + generated/reconstructed specialization
```

A `.sun` archive stores the information needed to recover model-specific weights relative to a compatible base model. Yozakura evaluates the complete system using peak RAM/VRAM, persistent storage, throughput, first-token latency, reconstruction quality, and energy consumption.

## Current implementation: `.sun`

A `.sun` (**Shared Universal Network**) archive contains:

- a reference to a compatible base model rather than another copy of its weights;
- shared low-rank factor prototypes grouped by module type;
- quantized per-layer residual factors;
- a deterministic manifest with checksums and evaluation metadata;
- non-executable tensor data stored with SafeTensors.

The target and base models must currently share the same architecture and matching projection tensor shapes.

## Install

```bash
pip install -e .
```

## Build a `.sun` archive

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

Tiered GPU/CPU/disk execution after eager host reconstruction:

```bash
yozakura run qwythos-9b.sun \
  --device auto \
  --max-memory 0=8GiB \
  --max-memory cpu=24GiB \
  --offload-folder .yozakura-offload \
  --prompt "Explain bounded-memory inference."
```

Fully out-of-core construction and execution:

```bash
yozakura run qwythos-9b.sun \
  --device out-of-core \
  --max-memory 0=8GiB \
  --max-memory cpu=16GiB \
  --offload-folder .yozakura-offload \
  --checkpoint-cache .yozakura-checkpoints \
  --workspace-mib 256 \
  --prompt "Explain out-of-core model execution."
```

The out-of-core path performs the following pipeline:

1. resolve the base SafeTensors checkpoint and its shard index;
2. read one base tensor at a time;
3. apply the matching SUN low-rank delta using bounded row chunks;
4. write a deterministic reconstructed SafeTensors cache;
5. instantiate the model structure on the `meta` device;
6. load weights directly into an inferred GPU/CPU/disk device map;
7. run generation through Accelerate offload hooks.

During reconstruction, peak model-weight memory is bounded by the current base tensor, the low-rank factors, and the configured SUN workspace rather than the complete checkpoint size. The reconstructed checkpoint cache is reused on later runs.

## `.sun` v1 layout

```text
model.sun
├── manifest.json
└── tensors.safetensors
```

`manifest.json` records the format version, base and target model IDs, module mapping, rank, prototype count, evaluation metadata, and SHA-256 checksum of the tensor payload. `tensors.safetensors` contains only non-executable tensor data.

## Runtime modes

| Mode | Initial host RAM requirement | Runtime placement |
|---|---:|---|
| `cpu` | complete reconstructed model | CPU |
| `cuda` | complete reconstructed model | one GPU |
| `auto` | complete reconstructed model during initialization | GPU + CPU + disk |
| `out-of-core` | current tensor + bounded SUN workspace | GPU + CPU + disk |

## Roadmap

### Implemented

- deterministic `.sun` artifacts;
- quantized low-rank reconstruction;
- bounded reconstruction workspace;
- isolated memory, speed, and quality benchmarks;
- GPU/CPU/disk tiered placement;
- SafeTensors shard resolution;
- full out-of-core checkpoint reconstruction;
- `meta` initialization and direct tiered loading.

### Next performance work

- asynchronous base-shard and SUN-factor prefetch;
- larger reconstructed shard packing to reduce filesystem overhead;
- direct reconstruction into Accelerate offload storage without an intermediate cache;
- fused reconstruction and matrix-multiplication kernels;
- quantized base-model backends;
- paged or quantized KV caches.

### Broader compatibility

- architecture adapters beyond shape-identical descendants;
- multimodal processor plumbing;
- GGUF-compatible loading where practical;
- reusable universal bases and composable specialization modules.

## Current limitations

- the base checkpoint must be available in SafeTensors format;
- the out-of-core path creates a reconstructed checkpoint cache, requiring temporary persistent storage approximately equal to the reconstructed model size;
- first execution includes reconstruction I/O and is slower than subsequent cached runs;
- tensor-per-file cache layout prioritizes bounded memory over filesystem efficiency;
- fused on-the-fly hypernetwork kernels are not yet implemented;
- target and base architectures must currently be shape-compatible;
- multimodal and GGUF paths are not yet supported.

## Security and licensing

`.sun` never uses pickle and does not embed Python code. The out-of-core reader rejects pickle-based `.bin` checkpoint shards. Consumers still need access to the declared base model and must comply with the licenses of both the base and target models.

## Success criteria

Yozakura considers the memory objective met when a consumer machine can construct and execute a compatible model whose conventional checkpoint exceeds available RAM or VRAM, while remaining within declared memory budgets. Quality, latency, storage amplification, and energy use remain independent release criteria.
