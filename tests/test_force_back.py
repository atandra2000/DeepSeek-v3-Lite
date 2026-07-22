"""Unit tests for the Triton env-var force-back guard (`models/_triton_dispatch.py`).

Without `ENABLE_TRITON_KERNELS=1`, a config with `attn_impl: "triton"` or
`moe_dispatch: "triton_grouped"` must be force-backed at model construction
time with a single warning — never per-layer at first invocation
(AGENTS rule #7).
"""
import sys
from pathlib import Path

from models._triton_dispatch import (  # noqa: E402
    _DISPATCH,
    enforce_triton_env_var,
)


# -----------------------------------------------------------------------------
# Direct tests of the helper.
# -----------------------------------------------------------------------------
class TestEnforceTritonEnvVar:
    def test_force_back_attn_impl_when_env_var_missing(self, monkeypatch):
        """`attn_impl='triton'` with no env-var must be force-backed to 'sdpa'."""
        monkeypatch.delenv("ENABLE_TRITON_KERNELS", raising=False)
        cfg = {"attn_impl": "triton"}
        captured: list = []
        enforce_triton_env_var(cfg, captured.append)
        assert cfg["attn_impl"] == "sdpa"
        assert any("attn_impl='triton' -> 'sdpa'" in m for m in captured)

    def test_force_back_moe_dispatch_when_env_var_missing(self, monkeypatch):
        """`moe_dispatch='triton_grouped'` with no env-var must be force-backed to 'stacked'."""
        monkeypatch.delenv("ENABLE_TRITON_KERNELS", raising=False)
        cfg = {"moe_dispatch": "triton_grouped"}
        captured: list = []
        enforce_triton_env_var(cfg, captured.append)
        assert cfg["moe_dispatch"] == "stacked"
        assert any("moe_dispatch='triton_grouped' -> 'stacked'" in m for m in captured)

    def test_force_back_both_keys_when_env_var_missing(self, monkeypatch):
        """Both Triton keys set + no env-var must both be force-backed in one warning."""
        monkeypatch.delenv("ENABLE_TRITON_KERNELS", raising=False)
        cfg = {"attn_impl": "triton", "moe_dispatch": "triton_grouped"}
        captured: list = []
        enforce_triton_env_var(cfg, captured.append)
        assert cfg["attn_impl"] == "sdpa"
        assert cfg["moe_dispatch"] == "stacked"
        # One warning, mentioning both.
        assert len(captured) == 1
        assert "attn_impl" in captured[0] and "moe_dispatch" in captured[0]

    def test_pass_through_when_env_var_set(self, monkeypatch):
        """`ENABLE_TRITON_KERNELS=1` lets the Triton values reach the model."""
        monkeypatch.setenv("ENABLE_TRITON_KERNELS", "1")
        cfg = {"attn_impl": "triton", "moe_dispatch": "triton_grouped"}
        captured: list = []
        enforce_triton_env_var(cfg, captured.append)
        assert cfg["attn_impl"] == "triton"
        assert cfg["moe_dispatch"] == "triton_grouped"
        assert captured == []

    def test_pytorch_values_unchanged(self, monkeypatch):
        """A 'sdpa' / 'stacked' / 'manual' config must pass through unchanged for all env-var states."""
        for env_val in (None, "0", "1"):
            if env_val is None:
                monkeypatch.delenv("ENABLE_TRITON_KERNELS", raising=False)
            else:
                monkeypatch.setenv("ENABLE_TRITON_KERNELS", env_val)
            cfg = {"attn_impl": "sdpa", "moe_dispatch": "stacked"}
            captured: list = []
            enforce_triton_env_var(cfg, captured.append)
            assert cfg["attn_impl"] == "sdpa"
            assert cfg["moe_dispatch"] == "stacked"
            assert captured == []

    def test_manual_attn_impl_unchanged(self, monkeypatch):
        """`attn_impl='manual'` is a PyTorch fallback path, not Triton; must pass through."""
        monkeypatch.delenv("ENABLE_TRITON_KERNELS", raising=False)
        cfg = {"attn_impl": "manual"}
        captured: list = []
        enforce_triton_env_var(cfg, captured.append)
        assert cfg["attn_impl"] == "manual"
        assert captured == []

    def test_no_warning_when_no_triton_keys_present(self, monkeypatch):
        """A config with neither Triton key set must not produce a warning."""
        monkeypatch.delenv("ENABLE_TRITON_KERNELS", raising=False)
        cfg = {"attn_impl": "sdpa", "moe_dispatch": "stacked", "dim": 768}
        captured: list = []
        enforce_triton_env_var(cfg, captured.append)
        assert captured == []
        # The non-triton keys are preserved.
        assert cfg["dim"] == 768

    def test_helper_tables_are_consistent(self):
        """The single dispatch table must pair each (key, triton_val) with a different pytorch_val."""
        from models._triton_dispatch import _DISPATCH
        for (key, triton_val), pytorch_val in _DISPATCH.items():
            assert triton_val != pytorch_val, (
                f"PyTorch default for {key} must differ from its Triton value"
            )

# -----------------------------------------------------------------------------
# Integration: building a Transformer triggers the guard.
# -----------------------------------------------------------------------------
class TestTransformerDispatchGuard:
    def test_transformer_force_backs_attn_impl(self, monkeypatch, capsys):
        """`Transformer(attn_impl='triton')` with no env-var must force-back at construction."""
        monkeypatch.delenv("ENABLE_TRITON_KERNELS", raising=False)
        from models.transformer import Transformer
        cfg = {
            "vocab_size": 64, "dim": 32, "n_layers": 1, "n_heads": 2,
            "n_dense_layers": 1, "n_routed_experts": 4, "n_shared_experts": 1,
            "n_activated_experts": 2, "inter_dim": 64, "moe_inter_dim": 32,
            "kv_lora_rank": 16, "q_lora_rank": 0, "qk_nope_head_dim": 8,
            "qk_rope_head_dim": 4, "v_head_dim": 8, "max_seq_len": 16,
            "rope_theta": 10000.0, "rope_factor": 1.0, "mscale": 1.0,
            "dtype": "bf16", "attn_impl": "triton", "weight_tying": True,
        }
        m = Transformer(cfg)
        # The guard must have rewritten the config to PyTorch default.
        assert cfg["attn_impl"] == "sdpa"
        # And the model's per-layer MLA must have read the rewritten value.
        assert m.layers[0].attn.attn_impl == "sdpa"
        # The warning must have been printed exactly once.
        captured = capsys.readouterr()
        assert "attn_impl='triton' -> 'sdpa'" in captured.out

    def test_transformer_pass_through_when_env_var_set(self, monkeypatch, capsys):
        """`Transformer(attn_impl='triton')` with `ENABLE_TRITON_KERNELS=1` keeps the Triton value."""
        monkeypatch.setenv("ENABLE_TRITON_KERNELS", "1")
        from models.transformer import Transformer
        cfg = {
            "vocab_size": 64, "dim": 32, "n_layers": 1, "n_heads": 2,
            "n_dense_layers": 1, "n_routed_experts": 4, "n_shared_experts": 1,
            "n_activated_experts": 2, "inter_dim": 64, "moe_inter_dim": 32,
            "kv_lora_rank": 16, "q_lora_rank": 0, "qk_nope_head_dim": 8,
            "qk_rope_head_dim": 4, "v_head_dim": 8, "max_seq_len": 16,
            "rope_theta": 10000.0, "rope_factor": 1.0, "mscale": 1.0,
            "dtype": "bf16", "attn_impl": "triton", "weight_tying": True,
        }
        m = Transformer(cfg)
        assert cfg["attn_impl"] == "triton"
        assert m.layers[0].attn.attn_impl == "triton"
        captured = capsys.readouterr()
        assert "forcing" not in captured.out

    def test_transformer_pytorch_config_is_silent(self, monkeypatch, capsys):
        """A default 'sdpa' / 'stacked' config must build with no warning."""
        monkeypatch.delenv("ENABLE_TRITON_KERNELS", raising=False)
        from models.transformer import Transformer
        cfg = {
            "vocab_size": 64, "dim": 32, "n_layers": 1, "n_heads": 2,
            "n_dense_layers": 1, "n_routed_experts": 4, "n_shared_experts": 1,
            "n_activated_experts": 2, "inter_dim": 64, "moe_inter_dim": 32,
            "kv_lora_rank": 16, "q_lora_rank": 0, "qk_nope_head_dim": 8,
            "qk_rope_head_dim": 4, "v_head_dim": 8, "max_seq_len": 16,
            "rope_theta": 10000.0, "rope_factor": 1.0, "mscale": 1.0,
            "dtype": "bf16", "attn_impl": "sdpa", "moe_dispatch": "stacked",
            "weight_tying": True,
        }
        m = Transformer(cfg)
        captured = capsys.readouterr()
        assert "forcing" not in captured.out
        assert m.layers[0].attn.attn_impl == "sdpa"
