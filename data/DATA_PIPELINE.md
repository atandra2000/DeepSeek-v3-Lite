# Data pipeline — redirect

The canonical DeepSeek-V3-Lite data guide is **[docs/09_Data_Pipeline.md](../docs/09_Data_Pipeline.md)**.

The universal 8.0B-token pipeline implementation lives in the workspace sibling **`LLM/shared_data/`** (imported by `data/prepare_data.py` via `sys.path`). See `LLM/shared_data/README.md` for mixture, tokenization, and sharding details.

**Quick start:**

```bash
python3 data/prepare_data.py --stage pretrain
```
