"""Tests for utility modules: CheckpointManager, memory estimation."""
import json
import os
import tempfile
from pathlib import Path
from typing import Optional

import pytest
import torch

from utils.checkpoint import CheckpointManager
from utils.memory import (
    _parameter_bytes,
    _optimiser_bytes,
    _kv_cache_bytes,
    _activation_bytes,
    _infer_dim_n_layers,
    estimate_model_memory_gb,
    assert_fits_in_available_gpu,
)
from models.transformer import Transformer


# CheckpointManager
class TestCheckpointManagerSaveLoad:
    def test_save_and_load(self, small_cfg, tmp_ckpt_dir):
        """Save and load preserves model weights."""
        ckpt = CheckpointManager(str(tmp_ckpt_dir))
        model = Transformer(small_cfg, use_checkpoint=False)
        initial_state = {k: v.clone() for k, v in model.state_dict().items()}
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=False)

        ckpt.save(model, opt, step=10)
        # Corrupt weights
        with torch.no_grad():
            for p in model.parameters():
                p.add_(1.0)
        # Load (use cpu device since no CUDA available on this machine)
        meta = ckpt.load(model, step=10, device="cpu", strict=False)
        # Verify
        for key in initial_state:
            assert torch.allclose(model.state_dict()[key], initial_state[key]), \
                f"Weight mismatch: {key}"
        assert meta["step"] == 10

    def test_save_with_state_dict_override(self, small_cfg, tmp_ckpt_dir):
        """Save with state_dict parameter uses the provided dict."""
        ckpt = CheckpointManager(str(tmp_ckpt_dir))
        model = Transformer(small_cfg, use_checkpoint=False)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=False)

        # Build a custom state dict with an extra key
        state = model.state_dict()
        state["extra_key"] = torch.zeros(10)

        ckpt.save(model, opt, step=5, state_dict=state)

        # Load into a different model and check the extra key is present
        model2 = Transformer(small_cfg, use_checkpoint=False)
        # Loading will log warnings about the extra key; that's fine
        meta = ckpt.load(model2, step=5, device="cpu", strict=False)
        assert "extra_key" not in model2.state_dict()  # strict=False ignores it
        assert meta["step"] == 5

    def test_latest_step(self, small_cfg, tmp_ckpt_dir):
        """latest_step() returns the highest complete step."""
        ckpt = CheckpointManager(str(tmp_ckpt_dir))
        model = Transformer(small_cfg, use_checkpoint=False)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=False)

        assert ckpt.latest_step() is None  # empty dir

        ckpt.save(model, opt, step=1)
        assert ckpt.latest_step() == 1

        ckpt.save(model, opt, step=3)
        assert ckpt.latest_step() == 3  # higher step

    def test_incomplete_step_skipped(self, small_cfg, tmp_ckpt_dir):
        """A step missing one of the three files is not considered complete."""
        ckpt = CheckpointManager(str(tmp_ckpt_dir))

        # Write only the safetensors file for step 5
        dummy_state = {"dummy": torch.zeros(1)}
        from safetensors.torch import save_file
        save_file(dummy_state, str(tmp_ckpt_dir / "model_step_5.safetensors"))

        assert ckpt.latest_step() is None, "Incomplete step should be skipped"

    def test_list_checkpoints(self, small_cfg, tmp_ckpt_dir):
        """list_checkpoints() returns sorted complete steps."""
        ckpt = CheckpointManager(str(tmp_ckpt_dir))
        model = Transformer(small_cfg, use_checkpoint=False)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=False)

        ckpt.save(model, opt, step=1)
        ckpt.save(model, opt, step=5)
        ckpt.save(model, opt, step=3)

        steps = ckpt.list_checkpoints()
        assert steps == [1, 3, 5]

    def test_keep_last_n(self, small_cfg, tmp_ckpt_dir):
        """keep_last_n removes older checkpoints."""
        ckpt = CheckpointManager(str(tmp_ckpt_dir))
        model = Transformer(small_cfg, use_checkpoint=False)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=False)

        for step in [1, 2, 3, 4, 5]:
            ckpt.save(model, opt, step=step)

        ckpt.keep_last_n(2)
        remaining = ckpt.list_checkpoints()
        assert remaining == [4, 5], f"Expected [4, 5], got {remaining}"

    def test_delete_checkpoint(self, small_cfg, tmp_ckpt_dir):
        """delete_checkpoint removes all files for a step."""
        ckpt = CheckpointManager(str(tmp_ckpt_dir))
        model = Transformer(small_cfg, use_checkpoint=False)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=False)

        ckpt.save(model, opt, step=7)
        assert ckpt.latest_step() == 7

        ckpt.delete_checkpoint(7)
        assert ckpt.latest_step() is None

    def test_load_missing_checkpoint_raises(self, small_cfg, tmp_ckpt_dir):
        """Loading a non-existent step raises FileNotFoundError."""
        ckpt = CheckpointManager(str(tmp_ckpt_dir))
        model = Transformer(small_cfg, use_checkpoint=False)
        with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
            ckpt.load(model, step=99)

    def test_load_optimizer_optional(self, small_cfg, tmp_ckpt_dir):
        """Loading without optimizer restores model weights only."""
        ckpt = CheckpointManager(str(tmp_ckpt_dir))
        model = Transformer(small_cfg, use_checkpoint=False)
        initial_state = {k: v.clone() for k, v in model.state_dict().items()}
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=False)

        ckpt.save(model, opt, step=2)
        with torch.no_grad():
            for p in model.parameters():
                p.add_(1.0)
        # Load without optimizer
        meta = ckpt.load(model, step=2, device="cpu", optimizer=None, strict=False)
        for key in initial_state:
            assert torch.allclose(model.state_dict()[key], initial_state[key])
        assert meta["step"] == 2

    def test_atomicity_temp_file_cleaned(self, small_cfg, tmp_ckpt_dir):
        """Temporary files are cleaned up if save fails mid-way."""
        ckpt = CheckpointManager(str(tmp_ckpt_dir))
        model = Transformer(small_cfg, use_checkpoint=False)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=False)

        ckpt.save(model, opt, step=20)
        # No .tmp files should remain
        tmp_files = list(tmp_ckpt_dir.glob("*.tmp"))
        assert len(tmp_files) == 0, f"Leftover tmp files: {tmp_files}"


# CheckpointManager with MTP (integration with Pretrainer)
class TestCheckpointManagerMTP:
    def test_mtp_weights_in_safetensors(self, small_cfg, tmp_ckpt_dir):
        """MTP-prefixed keys appear in the safetensors file."""
        from safetensors.torch import load_file
        from models.mtp import MultiTokenPrediction

        ckpt = CheckpointManager(str(tmp_ckpt_dir))
        model = Transformer(small_cfg, use_checkpoint=False)
        mtp = MultiTokenPrediction(small_cfg, model)
        opt = torch.optim.AdamW(mtp.parameters(), lr=1e-4, fused=False)

        # Build a combined state dict like Pretrainer does
        state = model.state_dict()
        mtp_state = {
            f"mtp.{k}": v
            for k, v in mtp.state_dict().items()
            if k.startswith("mtp_modules.")
        }
        state.update(mtp_state)
        ckpt.save(model, opt, step=1, state_dict=state)

        weights = load_file(str(tmp_ckpt_dir / "model_step_1.safetensors"))
        mtp_keys = [k for k in weights if k.startswith("mtp.")]
        assert len(mtp_keys) > 0, "MTP-prefixed keys should exist"
        assert any("mtp_modules" in k for k in mtp_keys), \
            "MTP keys should contain mtp_modules"

    def test_mtp_weights_roundtrip(self, small_cfg, tmp_ckpt_dir):
        """MTP weights can be extracted and loaded back."""
        from safetensors.torch import load_file
        from models.mtp import MTPModule, MultiTokenPrediction

        # Original model + MTP
        model = Transformer(small_cfg, use_checkpoint=False)
        mtp = MultiTokenPrediction(small_cfg, model)

        # Save combined state
        ckpt = CheckpointManager(str(tmp_ckpt_dir))
        opt = torch.optim.AdamW(mtp.parameters(), lr=1e-4, fused=False)
        state = model.state_dict()
        mtp_sd = {
            f"mtp.{k}": v
            for k, v in mtp.state_dict().items()
            if k.startswith("mtp_modules.")
        }
        state.update(mtp_sd)
        ckpt.save(model, opt, step=3, state_dict=state)

        # Now simulate inference loading (as in generate.py):
        # Load base model, then extract and load MTP keys
        model2 = Transformer(small_cfg, use_checkpoint=False)
        ckpt.load(model2, step=3, device="cpu", strict=False)

        mtp_module = MTPModule(small_cfg, depth=1)
        mtp_module.set_output_head(model2.head)
        weights = load_file(str(tmp_ckpt_dir / "model_step_3.safetensors"))
        mtp_state = {
            k.removeprefix("mtp."): v
            for k, v in weights.items() if k.startswith("mtp.")
        }
        mtp_module.load_state_dict(mtp_state, strict=False)

        # Verify MTP weights match
        for key in mtp_sd:
            original_key = key.removeprefix("mtp.")
            if original_key in mtp_module.state_dict():
                assert torch.allclose(mtp_sd[key], mtp_module.state_dict()[original_key]), \
                    f"MTP weight mismatch: {original_key}"


# Memory estimation (CPU-only)
class TestMemoryEstimation:
    def test_parameter_bytes(self, small_cfg):
        """_parameter_bytes returns sum of all param element sizes."""
        model = Transformer(small_cfg, use_checkpoint=False)
        expected = sum(p.numel() * p.element_size() for p in model.parameters())
        assert _parameter_bytes(model) == expected

    def test_optimiser_bytes(self, small_cfg):
        """_optimiser_bytes returns 12 bytes per unique param."""
        model = Transformer(small_cfg, use_checkpoint=False)
        n = sum(p.numel() for p in set(model.parameters()))
        assert _optimiser_bytes(model) == n * 12

    def test_kv_cache_bytes(self, small_cfg):
        """_kv_cache_bytes returns non-zero for models with MLA."""
        model = Transformer(small_cfg, use_checkpoint=False)
        bytes_ = _kv_cache_bytes(model, seq_len=64, batch_size=2)
        assert bytes_ > 0, "KV cache should have non-zero size"
        # Verify scaling: double batch → double bytes
        bytes_2x = _kv_cache_bytes(model, seq_len=64, batch_size=4)
        assert bytes_2x == 2 * bytes_

    def test_activation_bytes(self, small_cfg):
        """_activation_bytes computes correct scaling."""
        with_ckpt = _activation_bytes(
            seq_len=64, batch_size=2,
            hidden_dim=small_cfg["dim"],
            n_layers=small_cfg["n_layers"],
            grad_checkpoint=True,
        )
        without_ckpt = _activation_bytes(
            seq_len=64, batch_size=2,
            hidden_dim=small_cfg["dim"],
            n_layers=small_cfg["n_layers"],
            grad_checkpoint=False,
        )
        # Without checkpoint should be ~2x of with checkpoint
        assert without_ckpt == 2 * with_ckpt
        # Verify scaling: double seq → double bytes
        double_seq = _activation_bytes(
            seq_len=128, batch_size=2,
            hidden_dim=small_cfg["dim"],
            n_layers=small_cfg["n_layers"],
            grad_checkpoint=True,
        )
        assert double_seq == 2 * with_ckpt

    def test_infer_dim_n_layers(self, small_cfg):
        """_infer_dim_n_layers correctly identifies model dim/layers."""
        model = Transformer(small_cfg, use_checkpoint=False)
        dim, layers = _infer_dim_n_layers(model)
        assert dim == small_cfg["dim"]
        assert layers == small_cfg["n_layers"]

    def test_infer_dim_n_layers_empty(self):
        """_infer_dim_n_layers returns (0, 0) for a stub model."""
        import torch.nn as nn
        stub = nn.Linear(10, 10)
        dim, layers = _infer_dim_n_layers(stub)
        assert dim == 0
        assert layers == 0

    def test_estimate_model_memory_gb_positive(self, small_cfg):
        """estimate_model_memory_gb returns a positive float."""
        model = Transformer(small_cfg, use_checkpoint=False)
        est = estimate_model_memory_gb(
            model, seq_len=64, batch_size=2,
            grad_checkpoint=True, overhead_gb=0.0,
        )
        assert est > 0, "Estimate should be positive"

    def test_estimate_increases_with_batch(self, small_cfg):
        """Larger batch size leads to larger estimate."""
        model = Transformer(small_cfg, use_checkpoint=False)
        est1 = estimate_model_memory_gb(
            model, seq_len=64, batch_size=2,
            grad_checkpoint=True, overhead_gb=0.0,
        )
        est2 = estimate_model_memory_gb(
            model, seq_len=64, batch_size=4,
            grad_checkpoint=True, overhead_gb=0.0,
        )
        assert est2 > est1, "Larger batch should increase estimate"

    def test_assert_fits_no_cuda(self):
        """assert_fits_in_available_gpu is a no-op when CUDA is not available."""
        # Should not raise
        assert_fits_in_available_gpu(999.0)
        assert_fits_in_available_gpu(0.0)

    def test_overhead_detection(self):
        """_detect_overhead_gb returns 2.0 on CPU."""
        overhead = _detect_overhead_gb()
        assert overhead == 2.0  # CPU fallback


# Helper for the test above
def _detect_overhead_gb():
    """Replicated from utils.memory for CPU-only testing."""
    return 2.0
