# Hugging Face model support

Yozakura does not maintain a brittle list of individual checkpoint names. Compatibility is resolved at runtime from the Hugging Face configuration, registered `AutoModel` task class, and matching repeated linear tensor paths between the base and target models.

## Resolution pipeline

1. Load `AutoConfig` without loading model weights.
2. Resolve a task adapter from `model_type`, `architectures`, and encoder-decoder flags.
3. Load both checkpoints with the same task-specific `AutoModel` class.
4. Intersect `nn.Linear` paths whose tensor shapes match exactly.
5. In `--modules auto` mode, select suffixes repeated at least twice, excluding one-off heads by default.
6. Store the resolved task, model types, selected module suffixes, and exact tensor paths in the `.sun` manifest.

Use:

```bash
yozakura probe meta-llama/Llama-3.1-8B-Instruct
yozakura build --base OWNER/BASE --target OWNER/TARGET --output model.sun --task auto --modules auto
```

## Covered task adapters

- causal language modeling
- encoder-decoder language modeling
- masked language modeling
- sequence classification
- image classification
- vision-to-sequence generation
- speech sequence-to-sequence generation
- generic base/embedding models

Known family hints include Llama, Mistral/Mixtral, Qwen, Gemma, Phi, GPT-2/NeoX, Falcon, BLOOM, MPT, OPT, T5/mT5, BART, Pegasus, BERT, RoBERTa, DeBERTa, ELECTRA, ViT, Swin, ConvNeXT, Whisper, CLIP/SigLIP, LLaVA and IDEFICS. The hints are not a support boundary: newly registered Transformers architectures can resolve through `AutoConfig` without a Yozakura release.

## Compatibility contract

A base/target pair is buildable only when:

- both resolve to the same task adapter;
- selected linear paths exist in both models;
- each selected weight has the same shape;
- the release gates for approximation error and distribution size pass;
- required upstream custom code is explicitly enabled with `--trust-remote-code`.

"All Hugging Face models" cannot be guaranteed literally. Repositories may contain arbitrary remote code, unsupported tensor types, non-Transformers runtimes, mixture routing that is not represented by ordinary linear layers, or incompatible base/target architectures. Unsupported cases should fail explicitly rather than emit an invalid `.sun` artifact.

## Requesting an adapter

Open the repository's **AI model support survey** issue form. Include the exact model ID, task, `yozakura probe` output, base/target relationship, runtime requirements, and license constraints. This produces structured data that can later drive a public support matrix and CI fixtures.
