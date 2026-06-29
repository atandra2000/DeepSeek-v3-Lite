# Data pipeline — `data/prepare_data.py`

## Tokenizer

**`deepseek-coder-v2-lite`** BPE tokenizer:

- `vocab_size = 100,018`
- `eos_token_id = 100,017`
- `pad_token_id = 100,016`
- Has `byte_fallback` tokens — the embedding dim **must** equal
  `vocab_size` (100,018). This is the unusual part vs other LLM projects
  and is enforced across the model, tests, and config.

## Universal pipeline

`data/prepare_data.py` is a thin shim that materialises a
project-local `data/data_config.yaml` with the DeepSeek tokenizer
settings, then delegates to the shared `shared_data.prepare_data.run_pipeline`:

- **4 stages**: download → clean (quality + SHA-256 dedup) → tokenize →
  pack.
- **50M-token uint32 shards**, EOS-separated at document boundaries.
- `manifest.json` with full provenance.
- `LLM_DATA_ROOT` env var can point all 5 LLM projects at a shared cache;
  shards are bit-identical except for token IDs (which depend on the
  tokenizer).

## Mixture (canonical 422M run)

Per `AGENTS.md` / `CONTEXT.md` the DeepSeek-V3 mixture is
`fineweb 1.0 / smollm 0.6 / code 0.3 / cosmo 0.2 / math 0.1 / openmath 0.1`
(weights sum to 1.0). The shared universal mixture (`UNIVERSAL_MIXTURE_PATH`)
is the source of truth; `--mixture` overrides it. Add new sources by
editing the mixture file and re-running — see `SKILLS.md` Skill 6.

## CLI

```
python data/prepare_data.py --stage pretrain \
    [--mixture PATH] [--data-config PATH] [--data-root PATH] \
    [--source NAME] [--skip-download] [--skip-clean] \
    [--skip-tokenize] [--skip-pack]
```

`--stage` currently only accepts `pretrain`.