# Yozakura benchmarks

Use `yozakura-benchmark` to measure a `.sun` artifact in a fresh subprocess. Isolating each run prevents the previous model's allocator state from contaminating peak RAM and VRAM measurements.

## Quick benchmark

```bash
yozakura-benchmark \
  --archive model.sun \
  --device cuda \
  --prompt "The future of open source AI is" \
  --warmup 1 \
  --runs 3 \
  --max-new-tokens 64 \
  --output benchmark.json
```

## Compare against the declared target checkpoint

```bash
yozakura-benchmark \
  --archive model.sun \
  --device cuda \
  --compare-target \
  --warmup 1 \
  --runs 3 \
  --max-new-tokens 64 \
  --output benchmark-with-target.json
```

The target and `.sun` model are loaded in separate child processes so they do not reside in memory simultaneously.

## Reported metrics

- `.sun` archive size and complete manifest
- model load time
- first-token latency
- mean and median generation latency
- generated tokens per second
- model parameter bytes at the selected dtype
- current and peak process RSS on Linux
- peak allocated CUDA memory
- prompt cross-entropy and perplexity
- greedy output token IDs and decoded output
- top-10 next-token logits

When `--compare-target` is enabled, the report also includes:

- greedy generation token agreement
- next-token top-10 Jaccard overlap
- `.sun` / target throughput ratio
- `.sun` / target load-time ratio
- `.sun` / target peak RAM and VRAM ratios
- prompt-loss delta

## Interpretation

A useful artifact should normally satisfy all of the following:

1. `archive.bytes` is materially smaller than redistributing the full target checkpoint.
2. `comparison.prompt_loss_delta` remains small on representative prompts or datasets.
3. `comparison.generation_token_agreement` is stable enough for the intended application.
4. `sun.tokens_per_second` does not regress beyond the deployment budget.
5. Peak RAM/VRAM fits the intended hardware.

A single prompt is a smoke test, not a quality evaluation. For release decisions, run the benchmark repeatedly over a fixed prompt corpus and aggregate the JSON reports. Perplexity should also be evaluated on a held-out dataset rather than only on the prompt used for generation.

## CPU example

```bash
yozakura-benchmark \
  --archive tiny.sun \
  --device cpu \
  --dtype fp32 \
  --runs 2 \
  --max-new-tokens 32
```

## Notes

- `--compare-target` downloads and loads the target model declared by the `.sun` manifest.
- The current text benchmark requires a frontend that returns `input_ids`.
- Linux RSS values come from `/proc/self/status`; they are `null` on platforms without that interface.
- CUDA peak memory is allocated memory reported by PyTorch, not total driver-reserved memory.
