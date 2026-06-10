# models/transformer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .mla import MultiHeadLatentAttention
from .moe import DeepSeekMoE


class SwiGLUFFN(nn.Module):
    """
    SwiGLU feed-forward network used in dense layers.
    FFN(x) = W2(silu(W1(x)) * W3(x))
    Architecturally identical to the Expert FFN in DeepSeekMoE so that dense and sparse layers are interchangeable.
    """

    def __init__(self, dim: int, inter_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim, bias=False)
        self.w2 = nn.Linear(inter_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, inter_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    """
    Single transformer block: pre-norm MLA attention + pre-norm SwiGLU/MoE FFN. The first `n_dense_layers` layers
    use a SwiGLUFFN; all subsequent layers use DeepSeekMoE. Layer 0 is always dense in the config.
    """

    def __init__(
        self,
        layer_id: int,
        config: dict,
    ):
        super().__init__()
        self.layer_id      = layer_id
        self.dim           = config["dim"]
        self.n_dense_layers = config["n_dense_layers"]

        self.attn_norm = nn.RMSNorm(self.dim, eps=1e-6)
        self.attn = MultiHeadLatentAttention(config, layer_id)

        self.ffn_norm = nn.RMSNorm(self.dim, eps=1e-6)
        if layer_id < self.n_dense_layers:
            self.ffn = SwiGLUFFN(self.dim, config["inter_dim"])
        else:
            self.ffn = DeepSeekMoE(config)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int = 0,
        mask: Optional[torch.Tensor] = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), start_pos, mask, use_cache)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class ParallelEmbedding(nn.Module):
    """
    Vocabulary embedding for single-GPU setup.
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int,
    ):
        super().__init__()
        self.vocab_size  = vocab_size
        self.dim         = dim

        self.weight = nn.Parameter(torch.empty(vocab_size, dim))
        nn.init.normal_(self.weight, std=0.006)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.embedding(x, self.weight)



class Transformer(nn.Module):
    """
    DeepSeek-V3-style Transformer: MLA attention, MoE FFN, optional MTP.

    Construction
    ------------
    Takes a config dict directly (not a file path) so the model can be instantiated from memory, from tests, or
    after hydrating a checkpoint's stored config — without touching the filesystem.

    Causal mask caching
    -------------------
    `_build_causal_mask` caches the most recently built mask.  Calls with the same (seqlen, device) are free;
    a new mask is only allocated when seqlen or device changes.

    KV cache lifecycle
    ------------------
    During generation the KV cache is populated by each MLA layer. Call `reset_cache()` between independent
    generation requests to avoid stale context bleed and to return VRAM to the allocator.

    Gradient checkpointing
    ----------------------
    Set `use_checkpoint=True` to wrap each TransformerBlock in `torch.utils.checkpoint.checkpoint`.
    Reduces peak activation memory by recomputing activations backward at ~33% extra FLOPs.
    """

    def __init__(
        self,
        config: dict,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        # Accept either a raw model-config dict or a nested {"model": {...}} dict
        model_cfg = config.get("model", config)

        self.use_checkpoint = use_checkpoint
        self.max_seq_len    = model_cfg["max_seq_len"]
        self.config         = model_cfg   # stored for downstream use (e.g. MTP)

        self.embed = ParallelEmbedding(
            model_cfg["vocab_size"], model_cfg["dim"]
        )

        self.layers = nn.ModuleList(
            [
                TransformerBlock(i, model_cfg)
                for i in range(model_cfg["n_layers"])
            ]
        )

        self.norm = nn.RMSNorm(model_cfg["dim"], eps=1e-6)

        # Output head: full-vocab, non-sharded.
        # Weight tying: head.weight shares the same storage as embed.weight,
        # saving vocab_size * dim parameters (~9M for 14K vocab × 640 dim).
        self.weight_tying = model_cfg.get("weight_tying", False)
        self.head = nn.Linear(model_cfg["dim"], model_cfg["vocab_size"], bias=False)
        if self.weight_tying:
            self.head.weight = self.embed.weight  # share parameter storage

        # Causal mask cache: avoids re-allocating (S, S) on every forward call.
        self._mask_cache: Optional[torch.Tensor] = None
        self._mask_seqlen: int = 0


    # ──────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────

    def _build_causal_mask(self, seqlen: int, device: torch.device) -> torch.Tensor:
        """
        Return additive causal mask (1, 1, seqlen, seqlen).
        0 for attended, -inf for future tokens. Cached by seqlen+device.
        """
        if (
            self._mask_cache is None
            or seqlen != self._mask_seqlen
            or self._mask_cache.device != device
        ):
            mask = torch.triu(
                torch.full((seqlen, seqlen), float("-inf"), device=device),
                diagonal=1,
            )
            self._mask_cache  = mask.unsqueeze(0).unsqueeze(0)   # (1, 1, S, S)
            self._mask_seqlen = seqlen
        return self._mask_cache

    def _run_layers(
        self,
        h: torch.Tensor,
        start_pos: int,
        mask: Optional[torch.Tensor],
        use_cache: bool,
    ) -> torch.Tensor:
        """
        Run all TransformerBlocks, with gradient checkpointing enabled for training.
        """
        for layer in self.layers:
            if self.use_checkpoint and self.training:
                def _block(h, layer=layer, sp=start_pos, m=mask, uc=use_cache):
                    return layer(h, sp, m, uc)
                h = torch.utils.checkpoint.checkpoint(
                    _block, h, use_reentrant=False
                )
            else:
                h = layer(h, start_pos, mask, use_cache)
        return h


    def reset_cache(self) -> None:
        """Clear the KV cache in all MLA layers."""
        for layer in self.layers:
            if hasattr(layer.attn, "reset_cache"):
                layer.attn.reset_cache()

    def moe_layers(self):
        """Iterate over all DeepSeekMoE layers in the model."""
        for layer in self.layers:
            if isinstance(layer.ffn, DeepSeekMoE):
                yield layer.ffn

    # ──────────────────────────────────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────────────────────────────────

    def forward(
        self,
        tokens: torch.Tensor,
        start_pos: int = 0,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            tokens:    (bsz, seqlen) token IDs
            start_pos: KV-cache offset. 0 for prefill/training, incremented by seqlen per decode.
            use_cache: passed through to each MLA layer.

        Returns:
            logits: (bsz, seqlen, vocab_size). Index [:, -1, :] for last-position logits.
        """
        bsz, seqlen = tokens.shape
        h    = self.embed(tokens)
        mask = self._build_causal_mask(seqlen, tokens.device) if seqlen > 1 else None
        h    = self._run_layers(h, start_pos, mask, use_cache)
        h    = self.norm(h)
        return self.head(h)   # (bsz, seqlen, vocab_size)

    def forward_with_hidden(
        self,
        tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning both logits and final normalised hidden states. Used by MultiTokenPrediction in training.

        Returns:
            logits:  (bsz, seqlen, vocab_size)
            h_norm:  (bsz, seqlen, dim)
        """
        bsz, seqlen = tokens.shape
        h    = self.embed(tokens)
        mask = self._build_causal_mask(seqlen, tokens.device) if seqlen > 1 else None
        h    = self._run_layers(h, 0, mask, False)
        h_norm = self.norm(h)
        logits = self.head(h_norm)
        return logits, h_norm

    # ──────────────────────────────────────────────────────────────────────
    # Generation
    # ──────────────────────────────────────────────────────────────────────

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 512,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 0,
    ) -> torch.Tensor:
        """
        Autoregressive generation with KV-cache, top-p and top-k sampling.

        Prefill: encode full prompt in one forward pass, populate KV cache.
        Decode: one token at a time, start_pos advances by 1 each step.
        Each step is O(seq_so_far) in attention but O(1) per layer.

        Args:
            input_ids:      (bsz, prompt_len) — prompt token IDs
            max_new_tokens: maximum number of tokens to generate
            temperature:    softmax temperature; 1.0 = no change,
                            values < 1.0 sharpen
            top_p:          nucleus sampling threshold (0, 1]; 1.0 = disabled
            top_k:          top-k filtering; 0 = disabled

        Returns:
            (bsz, seq_out) token IDs
        """
        if temperature < 0.0:
            raise ValueError(f"temperature must be >= 0, got {temperature}")

        was_training = self.training
        self.reset_cache()
        self.eval()

        bsz, prompt_len = input_ids.shape
        output          = input_ids.clone()

        # ── Prefill ────────────────────────────────────────────────────────

        # Encode the full prompt once; this populates the KV cache for all positions [0, prompt_len).
        prefill_logits = self.forward(
            output, start_pos=0, use_cache=True)   # (bsz, prompt_len, vocab)
        # The next token is sampled from the last prompt position's logits.
        next_logits = prefill_logits[:, -1, :]   # (bsz, vocab)

        # ── Decode ─────────────────────────────────────────────────────────

        for step in range(max_new_tokens):
            next_token = self._sample(next_logits, temperature, top_p, top_k)
            output     = torch.cat([output, next_token], dim=1)

            if output.size(1) >= self.max_seq_len:
                break

            # Decode step: one token in, one token out.
            # start_pos = prompt_len + step because the cache already holds positions [0, prompt_len + step).
            decode_pos    = prompt_len + step
            decode_logits = self.forward(
                next_token, start_pos=decode_pos, use_cache=True
            )   # (bsz, 1, vocab)
            next_logits   = decode_logits[:, -1, :]

        if was_training:
            self.train()
        return output

    @staticmethod
    def _sample(
        logits: torch.Tensor,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> torch.Tensor:
        """
        Sample next token from logits.

        Applies temperature scaling, optional top-k and top-p truncation.
        Temperature == 0 uses argmax (greedy).

        Args:
            logits: (bsz, vocab)
            temperature: softmax temperature
            top_p: nucleus threshold; 1.0 disables
            top_k: keep top-k only; 0 disables

        Returns:
            (bsz, 1) sampled token IDs
        """
        if temperature == 0.0:
            return logits.argmax(dim=-1, keepdim=True)

        logits = logits / temperature

        # Top-k: zero out all but the k highest logits
        if top_k > 0:
            # Keep only the top-k values; fill the rest with -inf
            kth_vals = logits.topk(min(top_k, logits.size(-1)), dim=-1)[0][:, -1:]
            logits   = logits.masked_fill(logits < kth_vals, float("-inf"))

        probs = torch.softmax(logits, dim=-1)

        # Top-p: truncate to the smallest set of tokens whose cumulative probability exceeds p, then renormalise.
        if top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
            cumulative               = sorted_probs.cumsum(dim=-1)
            # Remove tokens where cumulative prob BEFORE adding this token > top_p
            remove       = (cumulative - sorted_probs) > top_p
            # Use masked_fill rather than in-place mutation to avoid corrupting the autograd graph if called in a grad context
            sorted_probs = sorted_probs.masked_fill(remove, 0.0)
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp(min=1e-10)
            # Sample from the truncated distribution and map back to vocab indices
            sampled_idx  = torch.multinomial(sorted_probs, num_samples=1)
            next_token   = sorted_idx.gather(-1, sampled_idx)
        else:
            next_token = torch.multinomial(probs, num_samples=1)

        return next_token   # (bsz, 1)


# ── Utilities ──────────────────────────────────────────────────────────────────

def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Return (total_params, trainable_params).
    Counts each unique parameter tensor once (shared weights visited once).
    """
    seen       = set()
    total      = 0
    trainable  = 0
    for p in model.parameters():
        pid = id(p)
        if pid in seen:
            continue
        seen.add(pid)
        n = p.numel()
        total     += n
        if p.requires_grad:
            trainable += n
    return total, trainable
