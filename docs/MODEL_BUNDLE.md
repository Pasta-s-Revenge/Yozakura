# Self-contained model bundles

A Yozakura model bundle is an offline directory containing everything needed to reconstruct and run one model:

```text
my-model/
├── bundle.json
├── model.sun
├── base/       # complete base-model checkpoint and config
└── frontend/   # tokenizer / processor / configuration files
```

The Yozakura Python runtime is still required. The bundle removes the runtime dependency on Hugging Face Hub and external model repositories; it does not package Python, PyTorch, CUDA, or operating-system libraries.

## Create a bundle

```bash
yozakura bundle qwythos-9b.sun \
  --output qwythos-9b-standalone \
  --revision <commit-or-tag>
```

Use a pinned Hugging Face commit SHA for reproducible distribution. The command downloads the complete base checkpoint and only frontend/configuration files from the target repository. Existing output directories are rejected unless `--force` is specified.

## Run offline

```bash
yozakura run qwythos-9b-standalone \
  --device out-of-core \
  --prompt "Explain bounded-memory inference."
```

`run` accepts either a normal `.sun` file or a bundle directory. For a bundle, Yozakura rewrites the archive references to absolute local paths and enables local-only loading, so no model files are fetched from the network.

## Distribution properties

- Base weights, SUN delta, tokenizer, processor, and model configuration are colocated.
- The directory is portable across machines with a compatible Yozakura runtime.
- Model licenses still apply to every copied base and target artifact.
- Bundle size is approximately the full base checkpoint plus the `.sun` delta.
- Architectures requiring custom remote Python code may need those `.py` files from the model repositories and `--trust-remote-code`.
