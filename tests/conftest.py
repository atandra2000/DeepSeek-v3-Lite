"""Shared fixtures for DeepSeek-V3-Lite test suite (CPU-only, small configs)."""
import os
import tempfile
from pathlib import Path
from typing import Dict, Tuple

import pytest
import torch


@pytest.fixture(scope="session")
def cfg() -> Dict:
    """Minimal model config matching the 82M architecture (truncated for test speed)."""
    return {
        "vocab_size":          100018,
        "dim":                 640,
        "n_layers":            2,
        "n_heads":             10,
        "n_dense_layers":      1,
        "n_routed_experts":    8,
        "n_shared_experts":    1,
        "n_activated_experts": 2,
        "inter_dim":           1280,
        "moe_inter_dim":       320,
        "kv_lora_rank":        128,
        "q_lora_rank":         0,
        "qk_nope_head_dim":    48,
        "qk_rope_head_dim":    16,
        "v_head_dim":          64,
        "max_seq_len":         128,
        "rope_theta":          10000,
        "rope_factor":         1.0,
        "mscale":              1.0,
        "mtp_depth":           1,
        "mtp_loss_weight":     0.3,
        "dtype":               "bf16",
        "attn_impl":           "sdpa",
        "use_grouped":         "stacked",
        "weight_tying":        True,
    }


@pytest.fixture(scope="session")
def small_cfg() -> Dict:
    """Even smaller config for extremely fast component tests."""
    return {
        "vocab_size":          1024,
        "dim":                 64,
        "n_layers":            2,
        "n_heads":             4,
        "n_dense_layers":      1,
        "n_routed_experts":    4,
        "n_shared_experts":    1,
        "n_activated_experts": 2,
        "inter_dim":           128,
        "moe_inter_dim":       64,
        "kv_lora_rank":        16,
        "q_lora_rank":         0,
        "qk_nope_head_dim":    8,
        "qk_rope_head_dim":    4,
        "v_head_dim":          8,
        "max_seq_len":         64,
        "rope_theta":          10000,
        "rope_factor":         1.0,
        "mscale":              1.0,
        "mtp_depth":           1,
        "mtp_loss_weight":     0.3,
        "dtype":               "bf16",
        "attn_impl":           "sdpa",
        "use_grouped":         "stacked",
        "weight_tying":        True,
    }


@pytest.fixture(scope="session")
def device() -> torch.device:
    return torch.device("cpu")


@pytest.fixture(scope="session")
def nested_cfg(cfg) -> Dict:
    """Config nested under the 'model' key (YAML format)."""
    return {"model": cfg}


@pytest.fixture(scope="session")
def training_cfg(cfg) -> Dict:
    """Full training config dict as parsed from YAML."""
    return {
        "model": cfg,
        "training": {
            "micro_batch_size":            2,
            "gradient_accumulation_steps": 2,
            "total_steps":                 10,
            "warmup_steps":                2,
            "lr":                          8.0e-4,
            "min_lr_ratio":                0.05,
            "weight_decay":                0.1,
            "grad_clip":                   1.0,
            "grad_checkpoint":             False,
            "compile":                     False,
            "save_interval":               5,
            "log_interval":               2,
            "mup_lr":                      False,
            "nan_guard":                   False,
            "bias_update_speed":           0.001,
            "bias_update_every":           10,
            "save_dir":                    "/tmp/test_checkpoints",
            "balance_loss_alpha":          0.0,
        },
        "data": {
            "train_data_path": "/tmp/test_data.bin",
        },
    }


# Token fixtures
@pytest.fixture(scope="session")
def tokens(small_cfg, device) -> torch.Tensor:
    """Random token IDs in the valid vocab range."""
    bsz, seq = 2, small_cfg["max_seq_len"]
    return torch.randint(0, small_cfg["vocab_size"] - 1, (bsz, seq), device=device)


@pytest.fixture(scope="session")
def targets(small_cfg, tokens) -> torch.Tensor:
    """Shifted targets (same as tokens, which is fine for shape tests)."""
    return tokens.clone()


# Temporary directory for checkpoint tests
@pytest.fixture()
def tmp_ckpt_dir():
    """Yield a temporary directory and clean up after the test."""
    tmp = Path(tempfile.mkdtemp(prefix="test_ckpt_"))
    yield tmp
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)


# Training data helpers
@pytest.fixture()
def tmp_data_file():
    """Write a small packed-token .bin file and yield its path."""
    data = torch.randint(0, 1024, (512,), dtype=torch.long)
    tmp = tempfile.NamedTemporaryFile(suffix=".bin", delete=False)
    torch.save(data, tmp.name)
    tmp.close()
    yield tmp.name
    os.unlink(tmp.name)


@pytest.fixture()
def tmp_shard_dir():
    """Create a directory with small shard files and yield the dir path."""
    tmp = Path(tempfile.mkdtemp(prefix="test_shards_"))
    for i in range(3):
        shard = torch.randint(0, 1024, (128,), dtype=torch.long)
        torch.save(shard, tmp / f"shard_{i:05d}.bin")
    yield str(tmp)
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
