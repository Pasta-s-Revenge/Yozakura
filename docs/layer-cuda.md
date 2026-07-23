# Hybrid layer streaming with CUDA

Yozakura's `layer` runtime can combine a bounded CUDA budget with CPU and disk offload. This is useful when the complete reconstructed checkpoint does not fit in VRAM but keeping several Transformer blocks resident on the GPU is still possible.

```bash
yozakura run model.sun \
  --device layer \
  --dtype float16 \
  --max-memory 0=8GiB \
  --max-memory cpu=6GiB \
  --offload-folder .yozakura-offload \
  --checkpoint-cache .yozakura-checkpoints \
  --workspace-mib 256 \
  --prompt "Explain hybrid layer streaming."
```

The numeric device key selects a CUDA device. For example, `0=8GiB` gives GPU 0 an 8 GiB placement budget. The CPU budget controls how many additional modules may remain resident in host memory. Modules that exceed both budgets are disk-offloaded and loaded by Accelerate hooks for their forward pass.

## Choosing a runtime

Use `cuda` when the complete model fits in VRAM. It avoids device-transfer and file-I/O overhead.

Use `auto` when the model can be reconstructed eagerly in host memory and then distributed across GPU, CPU, and disk.

Use `layer` when host-memory residency must also remain bounded. Adding a CUDA budget makes this a hybrid mode rather than CPU-only layer streaming.

## Performance notes

Layer streaming is a capacity feature, not a latency optimization. Autoregressive generation traverses every Transformer block for every generated token, so disk-offloaded blocks can cause substantial I/O overhead. Increase the CUDA and CPU budgets as far as the machine permits before relying on disk offload.

The reconstructed checkpoint cache is reused across runs. Keep `--checkpoint-cache` on fast local storage and avoid deleting it between prompts. The current checkpoint layout may contain many small SafeTensors shards, so storage latency and metadata operations can dominate short generations.

## Offline bundle example

```bash
yozakura run ./model-bundle \
  --device layer \
  --dtype float16 \
  --max-memory 0=8GiB \
  --max-memory cpu=6GiB \
  --offload-folder ./offload \
  --checkpoint-cache ./checkpoints \
  --local-files-only \
  --skip-checksum \
  --max-new-tokens 64 \
  --prompt "Explain binary search briefly."
```
