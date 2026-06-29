# Inference — `inference/`

## `generate.py`

Interactive generation entry point.

- `load_config(path)` — parses YAML, requires a `model` section.
- `generate_tokens(model, input_ids, max_new_tokens, temperature, top_p, eos_token_id)`
  — thin wrapper over `model.generate` under `@torch.inference_mode()`.
- `generate_interactive(model, tokenizer, args, mtp_module=None)` — REPL:
  `/exit` quits, `/clear` resets the message history. Builds a
  `SpeculativeDecoder` when `args.use_speculative and mtp_module is not None`.
- `main()` — loads config + checkpoint (dir or file), optionally loads
  MTP weights from the `mtp.`-prefixed keys of the safetensors file,
  loads the tokenizer (defaults to
  `deepseek-ai/deepseek-coder-v2-lite`), and enters the interactive loop.

CLI: `--config`, `--checkpoint`, `--max_new_tokens`, `--temperature`,
`--top_p`, `--use_speculative`, `--acceptance_threshold` (default 0.8).

## `speculative.py`

`SpeculativeDecoder` — MTP-based speculative decoder. See
[mtp.md](mtp.md) for the algorithm and load-bearing cache-sharing
invariant. Expected ~0.8 acceptance, up to 2× throughput at
`draft_depth=2`.

CLI usage (from `SKILLS.md`):

```
python inference/speculative.py --checkpoint checkpoints/dsv3_step_50000.pt \
  --prompt "The capital of France is" --draft_depth 2 --acceptance 0.8
```