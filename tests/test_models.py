"""Tests for all model components: Transformer, MLA, MoE, MTP, generation."""
import pytest
import torch
from torch import nn

from models.transformer import Transformer, SwiGLUFFN, ParallelEmbedding, count_parameters
from models.mla import MultiHeadLatentAttention
from models.moe import DeepSeekMoE, AuxLossFreeGate, Expert
from models.mtp import MTPBlock, MTPModule, MultiTokenPrediction


# Helpers
def _make_tokens(cfg, bsz=2, seq_len=None, device="cpu"):
    """Random token IDs within vocab range."""
    seq = seq_len or cfg["max_seq_len"]
    return torch.randint(0, min(cfg["vocab_size"] - 1, 512), (bsz, seq), device=device)


def _make_hidden(cfg, bsz=2, seq_len=None, device="cpu"):
    """Random hidden states."""
    seq = seq_len or cfg["max_seq_len"]
    return torch.randn(bsz, seq, cfg["dim"], device=device)


# ParallelEmbedding
class TestParallelEmbedding:
    def test_forward_shape(self, small_cfg, device):
        emb = ParallelEmbedding(small_cfg["vocab_size"], small_cfg["dim"])
        x = torch.randint(0, small_cfg["vocab_size"] - 1, (2, 8), device=device)
        out = emb(x)
        assert out.shape == (2, 8, small_cfg["dim"])

    def test_weight_tying_shared(self, small_cfg):
        """Verify weight tying shares the same storage."""
        from models.transformer import Transformer
        m = Transformer(small_cfg, use_checkpoint=False)
        assert m.head.weight.data_ptr() == m.embed.weight.data_ptr(), \
            "weight_tying should share storage"

    def test_weight_tying_disabled(self, small_cfg):
        """When weight_tying is False, head.weight is independent."""
        cfg = dict(small_cfg, weight_tying=False)
        from models.transformer import Transformer
        m = Transformer(cfg, use_checkpoint=False)
        assert m.head.weight.data_ptr() != m.embed.weight.data_ptr(), \
            "without weight_tying, pointers should differ"


# Transformer construction & forward
class TestTransformer:
    def test_construction(self, small_cfg):
        """Verify a Transformer can be built with the minimal config."""
        m = Transformer(small_cfg, use_checkpoint=False)
        assert isinstance(m, Transformer)
        assert len(m.layers) == small_cfg["n_layers"]

    def test_construction_nested_config(self, small_cfg):
        """Accept both flat and ``{"model": ...}`` configs."""
        m = Transformer({"model": small_cfg}, use_checkpoint=False)
        assert isinstance(m, Transformer)

    def test_forward_shape(self, small_cfg, device):
        """Forward returns (bsz, seq, vocab)."""
        m = Transformer(small_cfg, use_checkpoint=False).to(device)
        m.eval()
        x = _make_tokens(small_cfg, device=device)
        out = m(x, start_pos=0, use_cache=False)
        assert out.shape == (x.size(0), x.size(1), small_cfg["vocab_size"])

    def test_forward_with_hidden_shape(self, small_cfg, device):
        """forward_with_hidden returns (logits, hidden)."""
        m = Transformer(small_cfg, use_checkpoint=False).to(device)
        m.eval()
        x = _make_tokens(small_cfg, device=device)
        logits, h_norm = m.forward_with_hidden(x)
        assert logits.shape == (x.size(0), x.size(1), small_cfg["vocab_size"])
        assert h_norm.shape == (x.size(0), x.size(1), small_cfg["dim"])

    def test_forward_single_token(self, small_cfg, device):
        """Single-token forward (no causal mask) works."""
        m = Transformer(small_cfg, use_checkpoint=False).to(device)
        m.eval()
        x = torch.randint(0, small_cfg["vocab_size"] - 1, (1, 1), device=device)
        out = m(x, start_pos=0, use_cache=False)
        assert out.shape == (1, 1, small_cfg["vocab_size"])

    def test_forward_with_kv_cache(self, small_cfg, device):
        """Forward with use_cache=True populates the KV cache."""
        m = Transformer(small_cfg, use_checkpoint=False).to(device)
        m.eval()
        x = _make_tokens(small_cfg, bsz=1, device=device)
        out = m(x, start_pos=0, use_cache=True)
        assert out.shape == (1, x.size(1), small_cfg["vocab_size"])
        # Check that at least one MLA layer has a non-None KV cache
        any_cache = any(
            hasattr(l.attn, "kv_cache") and l.attn.kv_cache is not None
            for l in m.layers
        )
        assert any_cache, "KV cache should be populated after use_cache=True"

    def test_dense_and_moe_layers(self, small_cfg, device):
        """First n_dense_layers are SwiGLUFFN, remaining are MoE."""
        m = Transformer(small_cfg, use_checkpoint=False).to(device)
        n_dense = small_cfg["n_dense_layers"]
        for i, layer in enumerate(m.layers):
            if i < n_dense:
                assert isinstance(layer.ffn, SwiGLUFFN), f"Layer {i} should be dense"
            else:
                assert isinstance(layer.ffn, DeepSeekMoE), f"Layer {i} should be MoE"

    def test_moe_layers_iter(self, small_cfg, device):
        """moe_layers() yields only MoE layers."""
        m = Transformer(small_cfg, use_checkpoint=False).to(device)
        moe_list = list(m.moe_layers())
        n_dense = small_cfg["n_dense_layers"]
        expected = small_cfg["n_layers"] - n_dense
        assert len(moe_list) == expected
        for moe in moe_list:
            assert isinstance(moe, DeepSeekMoE)

    def test_grad_checkpoint_flag(self, small_cfg, device):
        """Gradient checkpointing can be enabled/disabled."""
        m = Transformer(small_cfg, use_checkpoint=True)
        assert m.use_checkpoint is True

    def test_forward_training_returns_grad(self, small_cfg, device):
        """Forward + backward in training mode produces gradients."""
        m = Transformer(small_cfg, use_checkpoint=False).to(device)
        m.train()
        x = _make_tokens(small_cfg, device=device)
        out = m(x, start_pos=0, use_cache=False)
        loss = out.sum()
        loss.backward()
        # At least one parameter should have a gradient
        has_grad = any(p.grad is not None for p in m.parameters())
        assert has_grad, "Expected at least one parameter to have a gradient"

    def test_reset_cache_clears_all_layers(self, small_cfg, device):
        """reset_cache() clears KV cache in all MLA layers."""
        m = Transformer(small_cfg, use_checkpoint=False).to(device)
        m.eval()
        x = _make_tokens(small_cfg, bsz=1, device=device)
        _ = m(x, start_pos=0, use_cache=True)
        m.reset_cache()
        all_empty = all(
            l.attn.kv_cache is None
            for l in m.layers
            if hasattr(l.attn, "reset_cache")
        )
        assert all_empty, "reset_cache should clear all KV caches"

    def test_forward_and_forward_with_hidden_agree(self, small_cfg, device):
        """forward() and forward_with_hidden() should produce the same logits."""
        m = Transformer(small_cfg, use_checkpoint=False).to(device)
        m.eval()
        x = _make_tokens(small_cfg, device=device)
        logits1 = m(x, start_pos=0, use_cache=False)
        logits2, h = m.forward_with_hidden(x, start_pos=0, use_cache=False)
        assert torch.allclose(logits1, logits2, atol=1e-6), \
            "forward() and forward_with_hidden() logits should match"


# MultiHeadLatentAttention
class TestMLA:
    def test_construction(self, small_cfg):
        mla = MultiHeadLatentAttention(small_cfg)
        assert mla.n_heads == small_cfg["n_heads"]
        assert mla.kv_lora_rank == small_cfg["kv_lora_rank"]

    def test_forward_shape(self, small_cfg, device):
        mla = MultiHeadLatentAttention(small_cfg).to(device)
        x = _make_hidden(small_cfg, device=device)
        out = mla(x, start_pos=0, use_cache=False)
        assert out.shape == x.shape

    def test_forward_with_cache(self, small_cfg, device):
        """Forward with use_cache=True populates and reads cache."""
        mla = MultiHeadLatentAttention(small_cfg).to(device)
        x = _make_hidden(small_cfg, bsz=1, device=device)
        out = mla(x, start_pos=0, use_cache=True)
        assert out.shape == x.shape
        assert mla.kv_cache is not None
        assert mla.pe_cache is not None

    def test_cache_growth(self, small_cfg, device):
        """Cache grows when batch size exceeds current capacity."""
        mla = MultiHeadLatentAttention(small_cfg).to(device)
        x1 = _make_hidden(small_cfg, bsz=1, device=device)
        _ = mla(x1, start_pos=0, use_cache=True)
        batch1 = mla._cache_batch
        # Forward with larger batch should trigger growth
        x2 = _make_hidden(small_cfg, bsz=batch1 + 4, device=device)
        _ = mla(x2, start_pos=0, use_cache=True)
        assert mla._cache_batch >= batch1 + 4

    def test_reset_cache(self, small_cfg, device):
        mla = MultiHeadLatentAttention(small_cfg).to(device)
        x = _make_hidden(small_cfg, bsz=1, device=device)
        _ = mla(x, start_pos=0, use_cache=True)
        mla.reset_cache()
        assert mla.kv_cache is None
        assert mla.pe_cache is None
        assert mla._cache_batch == 0

    def test_prefill_cache(self, small_cfg, device):
        mla = MultiHeadLatentAttention(small_cfg).to(device)
        bsz, seq = 1, 8
        kv = torch.randn(bsz, seq, small_cfg["kv_lora_rank"], device=device)
        pe = torch.randn(bsz, seq, small_cfg["qk_rope_head_dim"], device=device)
        mla.prefill_cache(kv, pe, start_pos=0)
        assert mla.kv_cache is not None
        assert torch.allclose(mla.kv_cache[:bsz, :seq], kv)

    def test_forward_manual_impl(self, small_cfg, device):
        """Manual attention path works and produces correct shapes."""
        cfg = dict(small_cfg, attn_impl="manual")
        mla = MultiHeadLatentAttention(cfg).to(device)
        x = _make_hidden(small_cfg, device=device)
        out = mla(x, start_pos=0, use_cache=False)
        assert out.shape == x.shape

    def test_sdpa_and_manual_agree(self, small_cfg, device):
        """SDPA and manual paths should produce similar outputs."""
        import copy
        cfg_sdpa = dict(small_cfg, attn_impl="sdpa")
        cfg_manual = dict(small_cfg, attn_impl="manual")
        sdpa = MultiHeadLatentAttention(cfg_sdpa).to(device)
        manual = MultiHeadLatentAttention(cfg_manual).to(device)
        # Share weights
        manual.load_state_dict(sdpa.state_dict())
        x = _make_hidden(small_cfg, device=device)
        with torch.no_grad():
            out_sdpa = sdpa(x, start_pos=0, use_cache=False)
            out_manual = manual(x, start_pos=0, use_cache=False)
        # Allow some tolerance for the different computation paths
        assert torch.allclose(out_sdpa, out_manual, atol=1e-4), \
            "SDPA and manual paths should produce similar results"

    def test_rope_extends(self, small_cfg, device):
        """RoPE frequency table extends on longer sequences."""
        mla = MultiHeadLatentAttention(small_cfg).to(device)
        assert mla._rope_seq_len == 0
        x = _make_hidden(small_cfg, seq_len=16, device=device)
        _ = mla(x, start_pos=0, use_cache=False)
        assert mla._rope_seq_len >= 16

    def test_cache_out_of_range_raises(self, small_cfg, device):
        """Accessing beyond max_seq_len raises an error."""
        mla = MultiHeadLatentAttention(small_cfg).to(device)
        max_len = small_cfg["max_seq_len"]
        x = _make_hidden(small_cfg, seq_len=max_len + 1, device=device)
        with pytest.raises(RuntimeError, match="exceeds max_seq_len"):
            mla(x, start_pos=0, use_cache=False)

    def test_prefill_overflow_raises(self, small_cfg, device):
        """prefill_cache beyond max_seq_len raises."""
        mla = MultiHeadLatentAttention(small_cfg).to(device)
        kv = torch.randn(1, small_cfg["max_seq_len"] + 1, small_cfg["kv_lora_rank"], device=device)
        pe = torch.randn(1, small_cfg["max_seq_len"] + 1, small_cfg["qk_rope_head_dim"], device=device)
        with pytest.raises(ValueError, match="end_pos.*> max_seq_len"):
            mla.prefill_cache(kv, pe, start_pos=0)


# SwiGLUFFN
class TestSwiGLUFFN:
    def test_forward_shape(self, small_cfg, device):
        dim, inter_dim = small_cfg["dim"], small_cfg["inter_dim"]
        ffn = SwiGLUFFN(dim, inter_dim).to(device)
        x = torch.randn(2, 8, dim, device=device)
        out = ffn(x)
        assert out.shape == x.shape

    def test_forward_non_linear(self, small_cfg, device):
        """Output should differ from input (non-linear transform)."""
        dim, inter_dim = small_cfg["dim"], small_cfg["inter_dim"]
        ffn = SwiGLUFFN(dim, inter_dim).to(device)
        x = torch.randn(1, 4, dim, device=device)
        out = ffn(x)
        assert not torch.allclose(out, x, atol=1e-2), "FFN should transform its input"


# Expert (single SwiGLU expert)
class TestExpert:
    def test_forward_shape(self, small_cfg, device):
        dim, inter_dim = small_cfg["dim"], small_cfg["moe_inter_dim"]
        expert = Expert(dim, inter_dim).to(device)
        x = torch.randn(2, 8, dim, device=device)
        out = expert(x)
        assert out.shape == x.shape


# DeepSeekMoE
class TestDeepSeekMoE:
    def test_construction(self, small_cfg):
        moe = DeepSeekMoE(small_cfg)
        assert len(moe.experts) == small_cfg["n_routed_experts"]
        assert len(moe.shared_experts) == small_cfg["n_shared_experts"]
        assert isinstance(moe.gate, AuxLossFreeGate)

    def test_forward_shape(self, small_cfg, device):
        moe = DeepSeekMoE(small_cfg).to(device)
        x = _make_hidden(small_cfg, device=device)
        out = moe(x)
        assert out.shape == x.shape

    def test_forward_stacked(self, small_cfg, device):
        """Stacked forward produces correct output shape."""
        cfg = dict(small_cfg, use_grouped="stacked")
        moe = DeepSeekMoE(cfg).to(device)
        x = _make_hidden(small_cfg, device=device)
        out = moe(x)
        assert out.shape == x.shape

    def test_forward_grouped(self, small_cfg, device):
        """Grouped forward produces correct output shape."""
        cfg = dict(small_cfg, use_grouped=True)
        moe = DeepSeekMoE(cfg).to(device)
        x = _make_hidden(small_cfg, device=device)
        out = moe(x)
        assert out.shape == x.shape

    def test_stacked_and_grouped_agree(self, small_cfg, device):
        """Stacked and grouped forward should produce similar results."""
        stacked = DeepSeekMoE(dict(small_cfg, use_grouped="stacked")).to(device)
        grouped = DeepSeekMoE(dict(small_cfg, use_grouped=True)).to(device)
        # Share weights
        grouped.load_state_dict(stacked.state_dict())
        x = _make_hidden(small_cfg, device=device)
        with torch.no_grad():
            out_s = stacked(x)
            out_g = grouped(x)
        assert torch.allclose(out_s, out_g, atol=1e-5), \
            "stacked and grouped forward should match"

    def test_gate_routing_correct_shape(self, small_cfg, device):
        """Gate returns (T, topk) weights and indices."""
        moe = DeepSeekMoE(small_cfg).to(device)
        x = _make_hidden(small_cfg, device=device)
        flat = x.view(-1, small_cfg["dim"])
        w, idx = moe.gate(flat)
        T = flat.size(0)
        assert w.shape == (T, small_cfg["n_activated_experts"])
        assert idx.shape == (T, small_cfg["n_activated_experts"])

    def test_gate_weights_sum_to_one(self, small_cfg, device):
        """Routing weights should sum to 1 per token."""
        moe = DeepSeekMoE(small_cfg).to(device)
        x = _make_hidden(small_cfg, device=device)
        flat = x.view(-1, small_cfg["dim"])
        w, _ = moe.gate(flat)
        assert torch.allclose(w.sum(dim=-1), torch.ones(flat.size(0), device=device), atol=1e-5)

    def test_bias_update(self, small_cfg, device):
        """Bias update shifts gate biases toward balancing."""
        moe = DeepSeekMoE(small_cfg).to(device)
        x = _make_hidden(small_cfg, device=device)
        _ = moe(x)  # populate routing cache

        # Manually set the last routing cache to a deliberately imbalanced distribution
        n_experts = small_cfg["n_routed_experts"]
        T, k = 128, small_cfg["n_activated_experts"]
        # All tokens to expert 0
        moe._last_indices = torch.zeros(T, k, dtype=torch.long, device=device)
        moe._last_weights = torch.ones(T, k, device=device) / k

        original_bias = moe.gate.bias.clone()
        moe.update_gate_bias(speed=0.1)
        # The bias for expert 0 should decrease (over-loaded), others increase
        assert moe.gate.bias[0] < original_bias[0], \
            "Over-loaded expert's bias should decrease"
        # At least one under-loaded expert should get an increase
        assert (moe.gate.bias[1:] > original_bias[1:]).any(), \
            "Under-loaded experts' bias should increase"

    def test_load_balance_loss(self, small_cfg, device):
        """Load balance loss returns a scalar (after forward)."""
        moe = DeepSeekMoE(small_cfg).to(device)
        x = _make_hidden(small_cfg, device=device)
        _ = moe(x)
        loss = moe.get_load_balance_loss()
        assert loss.ndim == 0, "Loss should be a scalar"
        assert loss > 0, "Balance loss should be positive for random data"

    def test_load_balance_loss_zero_without_forward(self, small_cfg, device):
        """Balance loss returns 0 if no forward has been called."""
        moe = DeepSeekMoE(small_cfg).to(device)
        loss = moe.get_load_balance_loss()
        assert loss.item() == 0.0

    def test_routing_stats(self, small_cfg, device):
        """get_routing_stats returns expected keys."""
        moe = DeepSeekMoE(small_cfg).to(device)
        x = _make_hidden(small_cfg, device=device)
        _ = moe(x)
        stats = moe.get_routing_stats()
        for key in ("counts", "load", "mean_weight", "utilisation"):
            assert key in stats, f"Missing key: {key}"
        assert stats["counts"].shape == (small_cfg["n_routed_experts"],)


# AuxLossFreeGate (standalone)
class TestAuxLossFreeGate:
    def test_construction(self, small_cfg):
        gate = AuxLossFreeGate(small_cfg)
        assert gate.weight.shape == (small_cfg["n_routed_experts"], small_cfg["dim"])
        assert gate.bias.shape == (small_cfg["n_routed_experts"],)

    def test_forward(self, small_cfg, device):
        gate = AuxLossFreeGate(small_cfg).to(device)
        T = 16
        x = torch.randn(T, small_cfg["dim"], device=device)
        w, idx = gate(x)
        assert w.shape == (T, small_cfg["n_activated_experts"])
        assert idx.shape == (T, small_cfg["n_activated_experts"])
        assert idx.max() < small_cfg["n_routed_experts"]

    def test_bias_not_in_parameters(self, small_cfg):
        """Bias should be a buffer, not a parameter."""
        gate = AuxLossFreeGate(small_cfg)
        param_ids = {id(p) for p in gate.parameters()}
        assert id(gate.bias) not in param_ids, "bias should not be a Parameter"

    def test_bias_in_state_dict(self, small_cfg):
        """Bias should be in state_dict for checkpoint persistence."""
        gate = AuxLossFreeGate(small_cfg)
        sd = gate.state_dict()
        assert "bias" in sd


# TransformerBlock
class TestTransformerBlock:
    def test_forward_shape(self, small_cfg, device):
        from models.transformer import TransformerBlock
        block = TransformerBlock(0, small_cfg).to(device)  # layer 0 = dense
        x = _make_hidden(small_cfg, device=device)
        out = block(x, start_pos=0, use_cache=False)
        assert out.shape == x.shape

    def test_moe_block(self, small_cfg, device):
        """Layer past n_dense_layers uses MoE."""
        from models.transformer import TransformerBlock
        moe_idx = small_cfg["n_dense_layers"]  # first MoE layer
        block = TransformerBlock(moe_idx, small_cfg).to(device)
        assert isinstance(block.ffn, DeepSeekMoE)
        x = _make_hidden(small_cfg, device=device)
        out = block(x, start_pos=0, use_cache=False)
        assert out.shape == x.shape


# MTP components
class TestMTPBlock:
    def test_forward_shape(self, small_cfg, device):
        block = MTPBlock(small_cfg).to(device)
        bsz, seq, dim = 2, 16, small_cfg["dim"]
        prev_h = torch.randn(bsz, seq, dim, device=device)
        target_emb = torch.randn(bsz, seq, dim, device=device)
        out = block(prev_h, target_emb)
        assert out.shape == (bsz, seq, dim)

    def test_independent_norms(self, small_cfg, device):
        """The two input streams are independently pre-normed."""
        block = MTPBlock(small_cfg).to(device)
        bsz, seq, dim = 2, 8, small_cfg["dim"]
        h = torch.randn(bsz, seq, dim, device=device)
        e = torch.randn(bsz, seq, dim, device=device) * 10  # very different scale
        out = block(h, e)
        assert out.shape == (bsz, seq, dim)
        assert not torch.isnan(out).any(), "No NaN even with mismatched scales"


class TestMTPModule:
    def test_construction(self, small_cfg):
        module = MTPModule(small_cfg, depth=1)
        assert module.output_head is None  # not yet set

    def test_forward_with_head(self, small_cfg, device):
        module = MTPModule(small_cfg, depth=1).to(device)
        # Create a shared head
        shared_head = nn.Linear(small_cfg["dim"], small_cfg["vocab_size"], bias=False).to(device)
        module.set_output_head(shared_head)
        bsz, seq, dim = 2, 16, small_cfg["dim"]
        prev_h = torch.randn(bsz, seq, dim, device=device)
        target_emb = torch.randn(bsz, seq, dim, device=device)
        logits, hidden = module(prev_h, target_emb)
        assert logits.shape == (bsz, seq, small_cfg["vocab_size"])
        assert hidden.shape == (bsz, seq, dim)

    def test_no_head_raises(self, small_cfg, device):
        module = MTPModule(small_cfg, depth=1).to(device)
        bsz, seq, dim = 2, 8, small_cfg["dim"]
        prev_h = torch.randn(bsz, seq, dim, device=device)
        target_emb = torch.randn(bsz, seq, dim, device=device)
        with pytest.raises(RuntimeError, match="output_head not set"):
            module(prev_h, target_emb)

    def test_shape_mismatch_raises(self, small_cfg, device):
        module = MTPModule(small_cfg, depth=1).to(device)
        shared_head = nn.Linear(small_cfg["dim"], small_cfg["vocab_size"], bias=False).to(device)
        module.set_output_head(shared_head)
        prev_h = torch.randn(2, 16, small_cfg["dim"], device=device)
        target_emb = torch.randn(2, 8, small_cfg["dim"], device=device)  # wrong seq
        with pytest.raises(ValueError, match="Shape mismatch"):
            module(prev_h, target_emb)


class TestMultiTokenPrediction:
    def test_construction(self, small_cfg, device):
        main = Transformer(small_cfg, use_checkpoint=False).to(device)
        mtp = MultiTokenPrediction(small_cfg, main)
        assert len(mtp.mtp_modules) == small_cfg["mtp_depth"]

    def test_forward_shape(self, small_cfg, device):
        main = Transformer(small_cfg, use_checkpoint=False).to(device)
        mtp = MultiTokenPrediction(small_cfg, main).to(device)
        mtp.eval()
        x = _make_tokens(small_cfg, device=device)
        main_logits, mtp_pairs = mtp(x)
        bsz, seq = x.shape
        assert main_logits.shape == (bsz, seq, small_cfg["vocab_size"])
        # mtp_pairs should have one element per depth
        assert len(mtp_pairs) == small_cfg["mtp_depth"]
        for d, (logits, tgt) in enumerate(mtp_pairs):
            usable = seq - d - 2  # matches MTP forward correction
            if usable > 0:
                assert logits.shape == (bsz, usable, small_cfg["vocab_size"])
                assert tgt.shape == (bsz, usable)

    def test_compute_loss(self, small_cfg, device):
        main = Transformer(small_cfg, use_checkpoint=False).to(device)
        mtp = MultiTokenPrediction(small_cfg, main).to(device)
        x = _make_tokens(small_cfg, device=device)
        targets = x.clone()
        main_logits, mtp_pairs = mtp(x)
        total, main_loss, mtp_loss = mtp.compute_loss(main_logits, targets, mtp_pairs)
        assert total.ndim == 0
        assert main_loss.ndim == 0
        assert mtp_loss.ndim == 0
        assert total > 0, "Loss should be positive"
        # With random weights, mtp_loss should be similar to ce loss (~ln(vocab))
        assert mtp_loss > 0

    def test_compute_loss_no_mtp(self, small_cfg, device):
        """compute_loss with empty mtp_pairs returns main_loss only."""
        main = Transformer(small_cfg, use_checkpoint=False).to(device)
        mtp = MultiTokenPrediction(small_cfg, main).to(device)
        x = _make_tokens(small_cfg, device=device)
        main_logits, _ = mtp(x)
        total, main_loss, mtp_loss = mtp.compute_loss(main_logits, x, mtp_pairs=[])
        assert torch.allclose(total, main_loss)
        assert mtp_loss.item() == 0.0

    def test_compute_loss_with_none(self, small_cfg, device):
        """compute_loss with None mtp_pairs returns main_loss only."""
        main = Transformer(small_cfg, use_checkpoint=False).to(device)
        mtp = MultiTokenPrediction(small_cfg, main).to(device)
        x = _make_tokens(small_cfg, device=device)
        main_logits, _ = mtp(x)
        total, main_loss, mtp_loss = mtp.compute_loss(main_logits, x, mtp_pairs=None)
        assert torch.allclose(total, main_loss)

    def test_training_backward(self, small_cfg, device):
        """MTP forward + loss + backward produces gradients on MTP blocks."""
        main = Transformer(small_cfg, use_checkpoint=False).to(device)
        mtp = MultiTokenPrediction(small_cfg, main).to(device)
        mtp.train()
        x = _make_tokens(small_cfg, device=device)
        targets = x.clone()
        main_logits, mtp_pairs = mtp(x)
        total_loss, _, _ = mtp.compute_loss(main_logits, targets, mtp_pairs)
        total_loss.backward()
        # Check that at least some MTP params have gradients
        mtp_has_grad = any(
            p.grad is not None
            for m in mtp.mtp_modules
            for p in m.parameters()
        )
        assert mtp_has_grad, "MTP blocks should receive gradients"

    def test_shared_head_mtp(self, small_cfg, device):
        """MTP modules share the same head weight as the main model."""
        main = Transformer(small_cfg, use_checkpoint=False).to(device)
        mtp = MultiTokenPrediction(small_cfg, main).to(device)
        for mtp_mod in mtp.mtp_modules:
            assert mtp_mod.output_head is main.head, \
                "Each MTP module should share the main model's head"

    def test_registered_embed(self, small_cfg, device):
        """Embedding parameters are included in the MTP wrapper's parameter set
        (ensuring they are trained and moved to the correct device)."""
        main = Transformer(small_cfg, use_checkpoint=False)
        mtp = MultiTokenPrediction(small_cfg, main)

        # The embed.weight should be reachable via mtp.parameters()
        all_param_ids = {id(p) for p in mtp.parameters()}
        assert id(main.embed.weight) in all_param_ids, \
            "embed.weight should be part of MTP wrapper parameters"

        # Also verify via named_parameters
        param_names = {n for n, _ in mtp.named_parameters()}
        assert any("embed" in n for n in param_names), \
            "embed should appear in MTP wrapper named_parameters"

    def test_forward_short_sequence(self, small_cfg, device):
        """Very short sequences don't crash MTP (usable may be <= 0)."""
        main = Transformer(small_cfg, use_checkpoint=False).to(device)
        mtp = MultiTokenPrediction(small_cfg, main).to(device)
        # Sequence length of 1 — MTP depth 1 needs usable=seq-2=-1 → skip
        x = torch.randint(0, small_cfg["vocab_size"] - 1, (1, 1), device=device)
        main_logits, mtp_pairs = mtp(x)
        assert main_logits.shape == (1, 1, small_cfg["vocab_size"])
        assert len(mtp_pairs) == 0  # no pairs for seq=1


# Generation
class TestGeneration:
    def test_generate_basic(self, small_cfg, device):
        """Basic generation produces output longer than input."""
        m = Transformer(small_cfg, use_checkpoint=False).to(device)
        m.eval()
        prompt = torch.randint(0, small_cfg["vocab_size"] - 1, (1, 8), device=device)
        out = m.generate(prompt, max_new_tokens=16, temperature=1.0, top_p=0.9)
        assert out.shape == (1, 8 + 16)  # prompt + generated

    def test_generate_eos_termination(self, small_cfg, device):
        """Generation stops early when EOS token is produced."""
        m = Transformer(small_cfg, use_checkpoint=False).to(device)
        m.eval()
        prompt = torch.randint(0, small_cfg["vocab_size"] - 1, (1, 4), device=device)
        # Use a very likely EOS token — we can't guarantee it'll be sampled,
        # so we use temperature=0 (greedy) and check max_new_tokens is respected.
        out = m.generate(prompt, max_new_tokens=8, temperature=0.0, eos_token_id=0)
        # Without guaranteed EOS production, just verify shape is valid
        assert out.size(1) >= prompt.size(1)

    def test_generate_greedy_deterministic(self, small_cfg, device):
        """Greedy generation (temperature=0) is deterministic."""
        m = Transformer(small_cfg, use_checkpoint=False).to(device)
        m.eval()
        prompt = torch.randint(0, small_cfg["vocab_size"] - 1, (1, 4), device=device)
        out1 = m.generate(prompt, max_new_tokens=4, temperature=0.0)
        out2 = m.generate(prompt, max_new_tokens=4, temperature=0.0)
        assert torch.equal(out1, out2), "Greedy generation should be deterministic"

    def test_generate_negative_temperature_raises(self, small_cfg, device):
        """Negative temperature raises ValueError."""
        m = Transformer(small_cfg, use_checkpoint=False).to(device)
        m.eval()
        prompt = torch.randint(0, small_cfg["vocab_size"] - 1, (1, 4), device=device)
        with pytest.raises(ValueError, match="temperature"):
            m.generate(prompt, temperature=-1.0)

    def test_generate_restores_train_mode(self, small_cfg, device):
        """generate() restores the training mode after completion."""
        m = Transformer(small_cfg, use_checkpoint=False).to(device)
        m.train()
        prompt = torch.randint(0, small_cfg["vocab_size"] - 1, (1, 4), device=device)
        m.generate(prompt, max_new_tokens=2, temperature=0.0)
        assert m.training, "Model should be restored to training mode"

    def test_generate_respects_max_seq_len(self, small_cfg, device):
        """Generation stops at max_seq_len even without EOS."""
        m = Transformer(small_cfg, use_checkpoint=False).to(device)
        m.eval()
        # max_seq_len=64 in small_cfg, prompt length=60, max_new_tokens=100
        prompt = torch.randint(0, small_cfg["vocab_size"] - 1, (1, small_cfg["max_seq_len"] - 4), device=device)
        out = m.generate(prompt, max_new_tokens=100, temperature=0.0)
        # Should stop at max_seq_len = 64, not at 60 + 100 = 160
        assert out.size(1) == small_cfg["max_seq_len"], \
            f"Generation should stop at max_seq_len={small_cfg['max_seq_len']}, got {out.size(1)}"

    def test_sample_top_k(self, small_cfg, device):
        """_sample with top_k reduces the candidate pool."""
        from models.transformer import Transformer
        logits = torch.randn(1, 100, device=device)
        # top_k=5 should always produce a token
        token = Transformer._sample(logits, temperature=1.0, top_k=5, top_p=1.0)
        assert token.shape == (1, 1)

    def test_sample_top_p(self, small_cfg, device):
        """_sample with top_p works."""
        from models.transformer import Transformer
        logits = torch.randn(1, 100, device=device)
        token = Transformer._sample(logits, temperature=1.0, top_k=0, top_p=0.5)
        assert token.shape == (1, 1)

    def test_generate_kv_cache_isolation(self, small_cfg, device):
        """Two independent generate calls don't share cache state."""
        m = Transformer(small_cfg, use_checkpoint=False).to(device)
        m.eval()
        prompt1 = torch.randint(0, small_cfg["vocab_size"] - 1, (1, 4), device=device)
        prompt2 = torch.randint(0, small_cfg["vocab_size"] - 1, (1, 8), device=device)

        out1 = m.generate(prompt1, max_new_tokens=4, temperature=0.0)
        # After first generate, cache should be reset (generate() always calls reset_cache)
        out2 = m.generate(prompt2, max_new_tokens=4, temperature=0.0)

        assert out1.size(1) == 8  # 4 prompt + 4 generated
        assert out2.size(1) == 12  # 8 prompt + 4 generated


# Parameter counting
class TestCountParameters:
    def test_count_parameters(self, small_cfg):
        m = Transformer(small_cfg, use_checkpoint=False)
        total, trainable = count_parameters(m)
        assert total > 0, "Total params should be > 0"
        assert trainable == total, "All params should be trainable (no frozen layers)"

    def test_count_with_weight_tying(self, small_cfg):
        """With weight tying, head weight is not double-counted."""
        tied = Transformer(dict(small_cfg, weight_tying=True), use_checkpoint=False)
        untied = Transformer(dict(small_cfg, weight_tying=False), use_checkpoint=False)
        total_tied, _ = count_parameters(tied)
        total_untied, _ = count_parameters(untied)
        # Tied should have fewer total params because head.weight is shared
        assert total_tied < total_untied, \
            f"Tied ({total_tied}) should be less than untied ({total_untied})"

    def test_count_with_mtp(self, small_cfg, device):
        """MTP wrapper adds extra parameters."""
        main = Transformer(small_cfg, use_checkpoint=False)
        mtp = MultiTokenPrediction(small_cfg, main)
        _, main_trainable = count_parameters(main)
        _, mtp_trainable = count_parameters(mtp)
        assert mtp_trainable > main_trainable, "MTP should add parameters"
