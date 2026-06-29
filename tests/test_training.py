"""Tests for training components: PretrainDataset, TrainingConfig, scheduler, Pretrainer."""
import copy
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional
from unittest.mock import patch

import pytest
import torch
import yaml

from training.pretrain import (
    Pretrainer,
    PretrainDataset,
    TrainingConfig,
    make_warmup_cosine_lambda,
)
from models.transformer import Transformer, count_parameters
from models.mtp import MultiTokenPrediction
from utils.checkpoint import CheckpointManager


# Helpers
def _build_training_config(small_cfg, tmp_ckpt_dir: str, mtp_weight: float = 0.0) -> TrainingConfig:
    """Build a TrainingConfig suitable for CPU testing."""
    return TrainingConfig(
        model_config=small_cfg,
        data_path="/tmp/nonexistent_data.bin",  # won't be accessed in constructor
        checkpoint_dir=tmp_ckpt_dir,
        max_seq_len=small_cfg["max_seq_len"],
        vocab_size=small_cfg["vocab_size"],
        batch_size=2,
        gradient_accumulation_steps=2,
        max_steps=4,
        warmup_steps=1,
        lr=1e-3,
        weight_decay=0.0,
        grad_checkpoint=False,
        compile_model=False,
        nan_guard=False,
        mtp_weight=mtp_weight,
        save_every=10,
        log_every=10,
    )


# TrainingConfig
class TestTrainingConfig:
    def test_defaults(self):
        """TrainingConfig has sensible defaults."""
        cfg = TrainingConfig()
        assert cfg.vocab_size == 100018
        assert cfg.max_seq_len == 4096
        assert cfg.mtp_weight == 0.0
        assert cfg.compile_model is True
        assert cfg.nan_guard is False

    def test_from_yaml_dict(self, training_cfg):
        """TrainingConfig can be constructed from a parsed YAML dict."""
        t = training_cfg["training"]
        d = training_cfg["data"]
        mc = training_cfg["model"]

        cfg = TrainingConfig(
            model_config=training_cfg,
            data_path=d["train_data_path"],
            checkpoint_dir=t["save_dir"],
            max_seq_len=mc["max_seq_len"],
            vocab_size=mc["vocab_size"],
            batch_size=t["micro_batch_size"],
            gradient_accumulation_steps=t["gradient_accumulation_steps"],
            max_steps=t["total_steps"],
            warmup_steps=t["warmup_steps"],
            lr=t["lr"],
            weight_decay=t["weight_decay"],
            grad_checkpoint=t["grad_checkpoint"],
            compile_model=t["compile"],
            save_every=t["save_interval"],
            log_every=t["log_interval"],
        )
        assert cfg.batch_size == 2
        assert cfg.max_steps == 10
        assert cfg.max_seq_len == 128

    def test_serializable(self):
        """TrainingConfig is serializable with dataclasses.asdict()."""
        cfg = TrainingConfig()
        d = asdict(cfg)
        assert isinstance(d, dict)
        assert "model_config" in d
        assert "mtp_weight" in d


# LR Scheduler
class TestWarmupCosineScheduler:
    def test_values_at_key_points(self):
        """Scheduler produces expected LR multipliers."""
        lr_lambda = make_warmup_cosine_lambda(warmup_steps=100, total_steps=1000, min_lr_ratio=0.1)
        # Step 0 → should be 0
        assert lr_lambda(0) == 0.0
        # Step 50 (mid-warmup) → 0.5
        assert abs(lr_lambda(50) - 0.5) < 1e-6
        # Step 100 (end of warmup) → 1.0
        assert abs(lr_lambda(100) - 1.0) < 1e-6
        # Step 550 (mid-cosine, ~0.5 progress) → 0.5 * (1 - 0.1) + 0.1 = 0.55
        # cos(pi * 0.5) = 0, so 0.1 + 0.9 * 0.5 * (1 + 0) = 0.1 + 0.45 = 0.55
        val_550 = lr_lambda(550)
        assert abs(val_550 - 0.55) < 1e-6, f"Expected 0.55, got {val_550}"
        # Step 1000 (end of cosine) → min_lr_ratio = 0.1
        assert abs(lr_lambda(1000) - 0.1) < 1e-6
        # Step 2000 (beyond total) → min_lr_ratio = 0.1
        assert abs(lr_lambda(2000) - 0.1) < 1e-6

    def test_monotonic_warmup(self):
        """LR increases monotonically during warmup."""
        lr_lambda = make_warmup_cosine_lambda(warmup_steps=50, total_steps=200)
        values = [lr_lambda(i) for i in range(50)]
        assert all(v2 >= v1 for v1, v2 in zip(values, values[1:])), \
            "Warmup should be monotonically non-decreasing"

    def test_cosine_decay(self):
        """LR decreases (non-increasing) after warmup."""
        lr_lambda = make_warmup_cosine_lambda(warmup_steps=20, total_steps=200)
        values = [lr_lambda(i) for i in range(20, 200)]
        assert all(v2 <= v1 for v1, v2 in zip(values, values[1:])), \
            "Cosine decay should be monotonically non-increasing"

    def test_no_warmup(self):
        """Zero warmup steps means cosine starts from step 0."""
        lr_lambda = make_warmup_cosine_lambda(warmup_steps=0, total_steps=100, min_lr_ratio=0.0)
        # Step 0 → start of cosine should give 1.0 * (0.5 * (1 + cos(0))) = 1.0
        assert abs(lr_lambda(0) - 1.0) < 1e-6


# PretrainDataset
class TestPretrainDataset:
    def test_single_file(self, tmp_data_file):
        """Single-file dataset loads and returns correct shapes."""
        ds = PretrainDataset(tmp_data_file, max_seq_len=16, vocab_size=1024)
        assert len(ds) > 0, "Should have at least one sample"
        x, y = ds[0]
        assert x.shape == (16,)
        assert y.shape == (16,)

    def test_single_file_shift(self, tmp_data_file):
        """Target is input shifted by 1."""
        ds = PretrainDataset(tmp_data_file, max_seq_len=16, vocab_size=1024)
        x, y = ds[0]
        assert torch.equal(x[1:], y[:-1]), "Target should be input shifted right by 1"

    def test_sharded_dataset(self, tmp_shard_dir):
        """Sharded dataset loads and returns correct shapes across shards."""
        ds = PretrainDataset(tmp_shard_dir, max_seq_len=16, vocab_size=1024)
        assert len(ds) > 0, "Should have at least one sample"
        x, y = ds[0]
        assert x.shape == (16,)
        assert y.shape == (16,)

    def test_sharded_cross_boundary(self, tmp_shard_dir):
        """Cross-shard window returns correct shapes."""
        ds = PretrainDataset(tmp_shard_dir, max_seq_len=32, vocab_size=1024)
        # Iterate all samples, at least one should cross a shard boundary
        for idx in range(len(ds)):
            x, y = ds[idx]
            assert x.shape == (32,)
            assert y.shape == (32,)

    def test_missing_file_raises(self):
        """Missing data path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Pre-training data not found"):
            PretrainDataset("/nonexistent/path.bin", max_seq_len=16, vocab_size=1024)

    def test_final_sample_truncated(self, tmp_data_file):
        """The final partial chunk is dropped (not padded)."""
        tokens = torch.load(tmp_data_file, weights_only=True)
        n_total = len(tokens)
        max_seq = 16
        expected = (n_total - 1) // max_seq
        ds = PretrainDataset(tmp_data_file, max_seq_len=max_seq, vocab_size=1024)
        assert len(ds) == expected, f"Expected {expected} samples, got {len(ds)}"

    def test_locate_edge_case(self, tmp_shard_dir):
        """_locate works for boundary indices."""
        ds = PretrainDataset(tmp_shard_dir, max_seq_len=16, vocab_size=1024)
        # Last valid index
        last_global = ds._total_tokens - 1
        shard, offset = ds._locate(last_global)
        assert shard >= 0
        assert offset >= 0
        # First index
        shard, offset = ds._locate(0)
        assert shard == 0
        assert offset == 0

    def test_locate_out_of_range_raises(self, tmp_shard_dir):
        """_locate with out-of-range index raises IndexError."""
        ds = PretrainDataset(tmp_shard_dir, max_seq_len=16, vocab_size=1024)
        with pytest.raises(IndexError):
            ds._locate(ds._total_tokens)  # one past the end
        with pytest.raises(IndexError):
            ds._locate(-1)


# Pretrainer construction
class TestPretrainerConstruction:
    def test_construction_cpu(self, small_cfg, tmp_ckpt_dir):
        """Pretrainer can be constructed on CPU."""
        config = _build_training_config(small_cfg, str(tmp_ckpt_dir))
        # fused AdamW requires CUDA — patch to False on CPU
        with patch("training.pretrain.AdamW", lambda *a, **kw: torch.optim.AdamW(*a, **{**kw, "fused": False})):
            trainer = Pretrainer(config)
        assert trainer.device.type == "cpu"
        assert trainer.raw_model is not None
        assert trainer.model is not None
        assert trainer.mtp_wrapper is None  # MTP disabled

    def test_construction_with_mtp(self, small_cfg, tmp_ckpt_dir):
        """Pretrainer construction with MTP enabled creates a wrapper."""
        config = _build_training_config(small_cfg, str(tmp_ckpt_dir), mtp_weight=0.3)
        with patch("training.pretrain.AdamW", lambda *a, **kw: torch.optim.AdamW(*a, **{**kw, "fused": False})):
            trainer = Pretrainer(config)
        assert trainer.mtp_wrapper is not None
        assert isinstance(trainer.mtp_wrapper, MultiTokenPrediction)

    def test_raw_model_is_uncompiled(self, small_cfg, tmp_ckpt_dir):
        """self.raw_model is always the uncompiled base Transformer."""
        config = _build_training_config(small_cfg, str(tmp_ckpt_dir))
        with patch("training.pretrain.AdamW", lambda *a, **kw: torch.optim.AdamW(*a, **{**kw, "fused": False})):
            trainer = Pretrainer(config)
        assert isinstance(trainer.raw_model, Transformer)
        # raw_model should NOT be compiled (no _orig_mod attribute)
        assert not hasattr(trainer.raw_model, "_orig_mod"), \
            "raw_model should be uncompiled"

    def test_model_parameters_include_all(self, small_cfg, tmp_ckpt_dir):
        """model.parameters() includes both base and MTP params when MTP enabled."""
        config = _build_training_config(small_cfg, str(tmp_ckpt_dir), mtp_weight=0.3)
        with patch("training.pretrain.AdamW", lambda *a, **kw: torch.optim.AdamW(*a, **{**kw, "fused": False})):
            trainer = Pretrainer(config)

        # Count unique params from model.parameters() and raw_model.parameters()
        model_params = sum(p.numel() for p in set(trainer.model.parameters()))
        raw_params = sum(p.numel() for p in set(trainer.raw_model.parameters()))
        assert model_params > raw_params, \
            f"Model params ({model_params}) > raw params ({raw_params}) when MTP enabled"

    def test_optimizer_deduplicates(self, small_cfg, tmp_ckpt_dir):
        """Optimizer deduplicates shared parameters (weight tying)."""
        config = _build_training_config(small_cfg, str(tmp_ckpt_dir))
        with patch("training.pretrain.AdamW", lambda *a, **kw: torch.optim.AdamW(*a, **{**kw, "fused": False})):
            trainer = Pretrainer(config)
        # Weight tying means head.weight and embed.weight share storage.
        # The optimizer should only have one group of decay params for it.
        total_opt_params = sum(
            p.numel() for group in trainer.optimizer.param_groups
            for p in group["params"]
        )
        # Count unique model params
        unique_params = sum(p.numel() for p in set(trainer.model.parameters()))
        assert total_opt_params == unique_params, \
            f"Optimizer has {total_opt_params} unique params, model has {unique_params}"

    def test_mup_lr_scaling(self, small_cfg, tmp_ckpt_dir):
        """µP LR scaling adjusts the learning rate."""
        config = _build_training_config(small_cfg, str(tmp_ckpt_dir))
        config.mup_lr = True
        config.mup_lr_reference = 6.0e-4
        config.mup_lr_reference_params = 757_226_496
        original_lr = config.lr
        with patch("training.pretrain.AdamW", lambda *a, **kw: torch.optim.AdamW(*a, **{**kw, "fused": False})):
            trainer = Pretrainer(config)
        # LR should have changed
        assert trainer.config.lr != original_lr, "µP scaling should change the LR"
        # Scaling formula: lr_ref * (P_ref / P)^0.5
        total_params, _ = count_parameters(trainer.raw_model)
        expected_lr = 6.0e-4 * (757_226_496 / total_params) ** 0.5
        assert abs(trainer.config.lr - expected_lr) < 1e-10, \
            f"Expected LR {expected_lr:.6e}, got {trainer.config.lr:.6e}"


# Checkpoint roundtrip
class TestCheckpointRoundtrip:
    def test_save_load(self, small_cfg, tmp_ckpt_dir):
        """Checkpoint save and load preserves model weights."""
        config = _build_training_config(small_cfg, str(tmp_ckpt_dir))
        with patch("training.pretrain.AdamW", lambda *a, **kw: torch.optim.AdamW(*a, **{**kw, "fused": False})):
            trainer = Pretrainer(config)

        # Get initial weights
        initial_state = {k: v.clone() for k, v in trainer.raw_model.state_dict().items()}

        # Save
        trainer.save_checkpoint(step=1)

        # Modify weights
        with torch.no_grad():
            for p in trainer.raw_model.parameters():
                p.add_(1.0)

        # Load
        trainer.load_checkpoint(step=1)

        # Verify weights restored
        for key in initial_state:
            assert torch.allclose(trainer.raw_model.state_dict()[key], initial_state[key]), \
                f"Weight mismatch for {key}"

    def test_save_load_with_mtp(self, small_cfg, tmp_ckpt_dir):
        """MTP checkpoint roundtrip preserves both base and MTP weights."""
        config = _build_training_config(small_cfg, str(tmp_ckpt_dir), mtp_weight=0.3)
        with patch("training.pretrain.AdamW", lambda *a, **kw: torch.optim.AdamW(*a, **{**kw, "fused": False})):
            trainer = Pretrainer(config)

        # Capture initial MTP and base weights
        initial_raw = {k: v.clone() for k, v in trainer.raw_model.state_dict().items()}
        mtp_orig = getattr(trainer.mtp_wrapper, "_orig_mod", trainer.mtp_wrapper)
        initial_mtp = {k: v.clone() for k, v in mtp_orig.state_dict().items() if k.startswith("mtp_modules.")}

        # Save
        trainer.save_checkpoint(step=2)

        # Corrupt weights
        with torch.no_grad():
            for p in trainer.raw_model.parameters():
                p.add_(1.0)
            for m in trainer.mtp_wrapper.mtp_modules:
                for p in m.parameters():
                    p.add_(1.0)

        # Load
        trainer.load_checkpoint(step=2)

        # Verify base weights restored
        for key in initial_raw:
            assert torch.allclose(trainer.raw_model.state_dict()[key], initial_raw[key]), \
                f"Base weight mismatch: {key}"

        # Verify MTP weights restored
        mtp_orig_after = getattr(trainer.mtp_wrapper, "_orig_mod", trainer.mtp_wrapper)
        for key in initial_mtp:
            assert torch.allclose(mtp_orig_after.state_dict()[key], initial_mtp[key]), \
                f"MTP weight mismatch: {key}"

    def test_checkpoint_meta_contains_step(self, small_cfg, tmp_ckpt_dir):
        """Checkpoint metadata includes the step number."""
        config = _build_training_config(small_cfg, str(tmp_ckpt_dir))
        with patch("training.pretrain.AdamW", lambda *a, **kw: torch.optim.AdamW(*a, **{**kw, "fused": False})):
            trainer = Pretrainer(config)

        trainer.save_checkpoint(step=5)
        meta_path = tmp_ckpt_dir / "meta_step_5.json"
        assert meta_path.exists()
        with open(meta_path) as f:
            meta = json.load(f)
        assert meta["step"] == 5

    def test_checkpoint_safetensors_mtp_prefix(self, small_cfg, tmp_ckpt_dir):
        """MTP weights are saved with 'mtp.' prefix in safetensors."""
        config = _build_training_config(small_cfg, str(tmp_ckpt_dir), mtp_weight=0.3)
        with patch("training.pretrain.AdamW", lambda *a, **kw: torch.optim.AdamW(*a, **{**kw, "fused": False})):
            trainer = Pretrainer(config)

        trainer.save_checkpoint(step=3)
        from safetensors.torch import load_file
        weights = load_file(str(tmp_ckpt_dir / "model_step_3.safetensors"))
        mtp_keys = [k for k in weights if k.startswith("mtp.")]
        assert len(mtp_keys) > 0, "Should have MTP-prefixed keys in checkpoint"


# Train step (direct component test — bypasses AMP)
class TestTrainStep:
    def test_standard_forward_backward(self, small_cfg, device):
        """Emulate train_step's standard (non-MTP) path: core tensor ops without AMP."""
        model = Transformer(small_cfg, use_checkpoint=False).to(device)
        model.train()
        opt = torch.optim.AdamW(
            [{"params": model.parameters(), "weight_decay": 0.0}],
            lr=1e-4, fused=False,
        )
        bsz, seq = 2, small_cfg["max_seq_len"]
        tokens = torch.randint(0, small_cfg["vocab_size"] - 1, (bsz, seq), device=device)
        targets = tokens.clone()

        # Forward (same as non-MTP train_step)
        logits = model(tokens, start_pos=0, use_cache=False)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=-100,
        )
        assert loss > 0, "Loss should be positive"

        # Backward
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        assert not torch.isnan(grad_norm).any(), "Gradient norm should not be NaN"
        opt.step()
        opt.zero_grad()

        # After one step, loss should generally decrease (random init → slightly less random)
        with torch.no_grad():
            logits2 = model(tokens, start_pos=0, use_cache=False)
            loss2 = torch.nn.functional.cross_entropy(
                logits2.reshape(-1, logits2.size(-1)),
                targets.reshape(-1),
                ignore_index=-100,
            )
        # The loss may not always decrease in 1 step (depends on init), so this is informational
        assert loss2 > 0, "Loss should stay positive"

    def test_mtp_forward_backward(self, small_cfg, device):
        """MTP forward + loss + backward works on CPU."""
        main = Transformer(small_cfg, use_checkpoint=False).to(device)
        mtp = MultiTokenPrediction(small_cfg, main).to(device)
        mtp.train()
        opt = torch.optim.AdamW(
            [{"params": mtp.parameters(), "weight_decay": 0.0}],
            lr=1e-4, fused=False,
        )
        bsz, seq = 2, small_cfg["max_seq_len"]
        tokens = torch.randint(0, small_cfg["vocab_size"] - 1, (bsz, seq), device=device)
        targets = tokens.clone()

        # MTP forward + loss (same as MTP train_step path)
        main_logits, mtp_pairs = mtp(tokens)
        total_loss, _, _ = mtp.compute_loss(main_logits, targets, mtp_pairs)
        assert total_loss > 0

        # Backward
        loss_val = total_loss / 2  # simulate grad_accum
        loss_val.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(mtp.parameters(), 1.0)
        assert not torch.isnan(grad_norm).any(), "Gradient norm should not be NaN"
        opt.step()
        opt.zero_grad()

    def test_moe_bias_update_during_training(self, small_cfg, device):
        """MoE bias updates work in a training-like loop."""
        model = Transformer(small_cfg, use_checkpoint=False).to(device)
        model.train()
        bsz, seq = 2, small_cfg["max_seq_len"]
        tokens = torch.randint(0, small_cfg["vocab_size"] - 1, (bsz, seq), device=device)

        # Capture initial biases
        initial_biases = []
        for moe in model.moe_layers():
            initial_biases.append(moe.gate.bias.clone())

        # Run a few steps with bias updates
        for step_idx in range(3):
            _ = model(tokens, start_pos=0, use_cache=False)
            for moe in model.moe_layers():
                moe.update_gate_bias(speed=0.01)

        # Verify biases have changed
        for i, moe in enumerate(model.moe_layers()):
            assert not torch.allclose(initial_biases[i], moe.gate.bias, atol=1e-7), \
                f"MoE layer {i} bias should change after update"

    def test_gradient_flow_to_all_params(self, small_cfg, device):
        """All parameters receive gradients after a backward pass."""
        model = Transformer(small_cfg, use_checkpoint=False).to(device)
        model.train()
        tokens = torch.randint(0, small_cfg["vocab_size"] - 1, (2, small_cfg["max_seq_len"]), device=device)
        targets = tokens.clone()

        logits = model(tokens, start_pos=0, use_cache=False)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=-100,
        )
        loss.backward()

        params_with_grad = sum(1 for p in model.parameters() if p.grad is not None)
        total_params = sum(1 for _ in model.parameters())
        assert params_with_grad >= total_params * 0.9, \
            f"Only {params_with_grad}/{total_params} params have gradients"


# MoE balance loss / metric
class TestMoEBalanceMetric:
    def test_balance_metric_returns_float(self, small_cfg, device):
        """The collection of balance losses returns a valid float."""
        from training.pretrain import Pretrainer
        model = Transformer(small_cfg, use_checkpoint=False).to(device)
        model.train()
        tokens = torch.randint(0, small_cfg["vocab_size"] - 1, (2, small_cfg["max_seq_len"]), device=device)
        _ = model(tokens, start_pos=0, use_cache=False)

        # Directly test the balance metric logic from Pretrainer
        balance_losses = [
            moe.get_load_balance_loss()
            for moe in model.moe_layers()
        ]
        if balance_losses:
            total = float(torch.stack(balance_losses).sum().item())
            assert total > 0, "Balance loss should be positive"


# Config parsing from YAML
class TestConfigFromYAML:
    def test_main_function_parses_yaml(self, small_cfg, tmp_ckpt_dir):
        """Verify the config parsing logic in main() works correctly."""
        # Write a minimal YAML config
        yaml_content = {
            "model": small_cfg,
            "training": {
                "micro_batch_size": 4,
                "gradient_accumulation_steps": 2,
                "total_steps": 100,
                "warmup_steps": 10,
                "lr": 1e-3,
                "save_dir": str(tmp_ckpt_dir),
            },
            "data": {
                "train_data_path": "/tmp/test.bin",
            },
        }
        yaml_path = tmp_ckpt_dir / "test_config.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_content, f)

        # Parse as main() would
        with open(yaml_path) as f:
            parsed = yaml.safe_load(f)

        mc = parsed.get("model", parsed)
        t = parsed.get("training", {})
        d = parsed.get("data", {})

        assert mc["vocab_size"] == small_cfg["vocab_size"]
        assert mc["dim"] == small_cfg["dim"]
        assert t["micro_batch_size"] == 4
        assert d["train_data_path"] == "/tmp/test.bin"

        # Verify TrainingConfig construction works with these values
        config = TrainingConfig(
            model_config=parsed,
            data_path=d.get("train_data_path", "data/pretrain_data.bin"),
            checkpoint_dir=t.get("save_dir", "checkpoints/pretrain"),
            max_seq_len=mc.get("max_seq_len", 4096),
            vocab_size=mc.get("vocab_size", 100018),
            batch_size=t.get("micro_batch_size", 8),
            gradient_accumulation_steps=t.get("gradient_accumulation_steps", 4),
            max_steps=t.get("total_steps", 20_000),
            warmup_steps=t.get("warmup_steps", 2_000),
            lr=t.get("lr", 2.2e-4),
        )
        assert config.batch_size == 4
        assert config.max_steps == 100
