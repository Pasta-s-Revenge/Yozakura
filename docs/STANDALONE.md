# Standalone distribution

Yozakura can be packaged as a self-contained application directory. End users do not need to install Python or the project's Python dependencies.

The bundle uses PyInstaller's one-folder layout. A single-file executable is intentionally avoided because PyTorch, Transformers, native libraries, and optional accelerator backends are large and are more reliable when kept as adjacent files.

## Build locally

Use Python 3.10 or newer on the target operating system:

```bash
python -m pip install --upgrade pip
python -m pip install ".[standalone]"
python scripts/build-standalone.py --clean
```

The result is written to:

```text
dist/yozakura/
```

Run it with:

```bash
./dist/yozakura/yozakura --help
./dist/yozakura/yozakura inspect model.sun
```

On Windows, use `dist\yozakura\yozakura.exe`.

## CI artifacts

The **Standalone bundles** workflow can be started manually from GitHub Actions. Tags matching `v*` also trigger it. It produces Linux x86-64 and Windows x86-64 artifact directories.

## Portability constraints

- Build separately for each operating system and CPU architecture.
- The default bundle follows the PyTorch wheel installed in the build environment.
- CUDA-enabled distribution requires building with a matching CUDA PyTorch wheel and compatible NVIDIA runtime libraries.
- Hugging Face model files and `.sun` archives are not embedded. They remain external data and can be downloaded or supplied locally at runtime.
- `--trust-remote-code` may load model-specific Python code that was not frozen into the bundle. Standalone releases should therefore prefer model architectures already supported by the bundled Transformers version.
