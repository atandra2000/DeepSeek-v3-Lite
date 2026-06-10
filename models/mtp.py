# models/mtp.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


class MTPBlock(nn.Module):
    """
    Single transformer-like block inside an MTP head (DeepSeek-V3 §2.4).

    Given h_prev from the previous depth and the embedding of the next
    target token e_target, the block:
      1. Independently pre-norms each input stream.
      2. Concatenates and projects to `dim` (fusion).
      3. Pre-norm causal self-attention residual.
      4. Pre-norm SwiGLU FFN residual.

    Independent pre-norms are critical: h_prev sits in well-scaled space
    while e_target is raw embedding; without them, magnitudes mismatch.

    Causal mask is cached and resized on demand to avoid reallocation.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.dim      = config["dim"]
        n_heads       = config["n_heads"]
        inter_dim     = config["inter_dim"]

        # Independent pre-norms for each input stream before fusion
        self.norm_h   = nn.RMSNorm(self.dim, eps=1e-6)   # for prev_hidden
        self.norm_e   = nn.RMSNorm(self.dim, eps=1e-6)   # for target_emb

        # Fusion: [norm(h) ‖ norm(e)] → dim
        self.proj     = nn.Linear(self.dim * 2, self.dim, bias=False)

        # Transformer sub-layers (pre-norm)
        self.norm_attn = nn.RMSNorm(self.dim, eps=1e-6)
        self.attn      = nn.MultiheadAttention(
            self.dim, n_heads, batch_first=True, bias=False
        )
        self.norm_ffn  = nn.RMSNorm(self.dim, eps=1e-6)
        self.w1        = nn.Linear(self.dim, inter_dim, bias=False)
        self.w2        = nn.Linear(inter_dim, self.dim, bias=False)
        self.w3        = nn.Linear(self.dim, inter_dim, bias=False)

        # Causal mask cache: (max_cached_seq, max_cached_seq)
        # Stored as a non-persistent buffer so it moves with the module but is not saved in checkpoints.
        self._causal_mask_size: int = 0
        self.register_buffer("_causal_mask", torch.empty(0, 0), persistent=False)

    def _get_causal_mask(self, seqlen: int, device: torch.device) -> torch.Tensor:
        """Return a (seqlen, seqlen) upper-triangular -inf mask, cached."""
        if seqlen > self._causal_mask_size or self._causal_mask.device != device:
            mask = torch.triu(
                torch.full((seqlen, seqlen), float("-inf"), device=device),diagonal=1)
            self._causal_mask      = mask
            self._causal_mask_size = seqlen
        return self._causal_mask[:seqlen, :seqlen]

    def forward(
        self,
        prev_hidden: torch.Tensor,
        target_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            prev_hidden: (bsz, seq, dim) — normalised hidden states from the
                         previous MTP depth (or from the main model at depth 0).
            target_emb:  (bsz, seq, dim) — embedding of the target tokens at
                         this depth (raw, unnormalised lookup output).

        Returns:
            updated hidden states (bsz, seq, dim)
        """
        # 1. Independent pre-norms then fuse
        fused   = self.proj(
            torch.cat([self.norm_h(prev_hidden), self.norm_e(target_emb)], dim=-1))   # (bsz, seq, dim)

        # 2. Pre-norm causal self-attention
        seqlen  = fused.size(1)
        mask    = self._get_causal_mask(seqlen, fused.device)
        attn_in = self.norm_attn(fused)
        attn_out, _ = self.attn(
            attn_in, attn_in, attn_in,
            attn_mask=mask, is_causal=False)   # mask is supplied explicitly
        fused   = fused + attn_out

        # 3. Pre-norm SwiGLU FFN
        ffn_in  = self.norm_ffn(fused)
        ffn_out = self.w2(F.silu(self.w1(ffn_in)) * self.w3(ffn_in))
        return fused + ffn_out


class MTPModule(nn.Module):
    """
    MTP head for prediction depth *d* (1-indexed).

    Predicts token t_{i+d} given:
      • prev_hidden: the hidden state at position i from depth d-1
      • target_emb:  the embedding of token t_{i+d-1}

    Output head
    -----------
    When `output_head` is None, the module raises at forward time.  The intended usage is for the caller
    (MultiTokenPrediction) to inject a shared weight reference via `set_output_head()` after construction, tying all
    MTP modules and the main LM head to the same projection. This matches the DeepSeek-V3 design.

    """

    def __init__(self, config: dict, depth: int = 1):
        super().__init__()
        self.depth      = depth
        self.dim        = config["dim"]
        self.vocab_size = config["vocab_size"]

        self.block      = MTPBlock(config)
        self.norm       = nn.RMSNorm(self.dim, eps=1e-6)

        # Placeholder; replaced by set_output_head() in MultiTokenPrediction
        self.output_head: Optional[nn.Linear] = None

    def set_output_head(self, head: nn.Linear) -> None:
        """
        Bind a shared output projection.  Must be called before forward().
        The head is stored as a plain attribute (not via add_module) so that its parameters are not double-counted.
        """
        self.output_head = head

    def forward(
        self,
        prev_hidden: torch.Tensor,
        target_emb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            prev_hidden: (bsz, seq, dim) — normalised hidden states
            target_emb:  (bsz, seq, dim) — raw target token embeddings

        Returns:
            logits: (bsz, seq, vocab_size)
            hidden: (bsz, seq, dim) — normalised hidden states for next depth
        """
        if self.output_head is None:
            raise RuntimeError(
                f"MTPModule(depth={self.depth}): output_head not set. "
                "Call set_output_head() before forward()."
            )
        if prev_hidden.shape != target_emb.shape:
            raise ValueError(
                f"Shape mismatch: prev_hidden {prev_hidden.shape} "
                f"vs target_emb {target_emb.shape}"
            )
        h      = self.block(prev_hidden, target_emb)
        h_norm = self.norm(h)
        logits = self.output_head(h_norm)
        return logits, h_norm


class MultiTokenPrediction(nn.Module):
    """
    Wraps a main Transformer model with D MTP prediction heads.

    Token alignment
    ---------------
    For a sequence of length S, positions are 0-indexed:

      Depth 0 (main model):  input  tokens[0..S-1]
                             target tokens[1..S]     (standard LM shift)
      MTP depth 1:           input  tokens[0..S-2]   (h from main, emb of tokens[1])
                             target tokens[2..S]
      MTP depth d:           input  tokens[0..S-d-1]
                             target tokens[d+1..S]

    At each depth the usable sequence shrinks by 1.  `forward()` returns aligned (logits, targets) pairs
    so `compute_loss()` never needs to re-derive offsets.

    Shared output head
    ------------------
    All MTP modules share the main model's `head` weight. This is injected via `MTPModule.set_output_head()`.
    No extra parameters are added beyond the MTPBlock weights.

    Shared embedding
    ----------------
    `self.embed` is registered as a proper submodule reference via `add_module` so that
    `.to()`, `.parameters()`, and `.state_dict()` on this wrapper all see it correctly — even if the main model's
    embedding is later replaced.
    """

    def __init__(self, config: dict, main_model: nn.Module):
        super().__init__()
        self.main_model = main_model

        # Accept flat config (mtp_depth / mtp_loss_weight) or
        # nested config (config["mtp"]["depth"] / ["weight"])
        if "mtp_depth" in config:
            self.depth      = config["mtp_depth"]
            self.mtp_weight = config.get("mtp_loss_weight", 0.3)
        else:
            mtp_cfg     = config.get("mtp", {})
            self.depth      = mtp_cfg.get("depth", 1)
            self.mtp_weight = mtp_cfg.get("weight", 0.3)

        model_cfg = config.get("model", config)

        self.mtp_modules = nn.ModuleList(
            [MTPModule(model_cfg, d + 1) for d in range(self.depth)]
        )

        # Register the shared embedding as a submodule so device/dtype moves and parameter iteration work correctly
        self.add_module("embed", main_model.embed)

        # Share the main model's output head with all MTP modules. Using set_output_head()
        # stores the reference as a plain attribute so the head's parameters are not double-counted.
        shared_head = main_model.head
        for mtp in self.mtp_modules:
            mtp.set_output_head(shared_head)

    def forward(
        self,
        tokens: torch.Tensor,
        start_pos: int = 0,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Run the main model and all MTP heads.

        Args:
            tokens:    (bsz, seq) token IDs
            start_pos: KV-cache offset (0 during training)

        Returns:
            main_logits:  (bsz, seq, vocab)
            mtp_pairs:    list of (logits, targets) per MTP depth, where
                          logits  is (bsz, seq-d-1, vocab) and
                          targets is (bsz, seq-d-1) — already aligned,
                          ready to pass directly to cross_entropy.
        """
        if tokens.dim() < 2:
            raise ValueError(f"Expected (bsz, seq) tokens, got shape {tokens.shape}")

        seq_len = tokens.size(1)

        main_logits, prev_h = self.main_model.forward_with_hidden(tokens)
        # prev_h: (bsz, seq, dim) — final-norm hidden states from main model

        mtp_pairs: List[Tuple[torch.Tensor, torch.Tensor]] = []

        for d, mtp in enumerate(self.mtp_modules):
            # At depth d, the usable sequence has length seq - d - 1.
            # prev_h covers positions [0, seq-d-1) from the previous depth.
            # target_emb is the embedding of tokens at positions [d+1, seq).
            # The prediction target is tokens at positions [d+2, seq).
            usable = seq_len - d - 1
            if usable <= 0:
                break
            # Hidden states: trim to usable length from the right
            h_in   = prev_h[:, :usable]                       # (bsz, usable, dim)
            # Target token embeddings: tokens[d+1 .. seq-1]
            emb_in = self.embed(tokens[:, d + 1 : d + 1 + usable])  # (bsz, usable, dim)
            # Prediction targets: tokens[d+2 .. seq]
            tgt    = tokens[:, d + 2 : d + 2 + usable]        # (bsz, usable)
            logits, hidden = mtp(h_in, emb_in)
            mtp_pairs.append((logits, tgt))
            prev_h = hidden   # pass normalised hidden states to next depth

        return main_logits, mtp_pairs

    def compute_loss(
        self,
        main_logits: torch.Tensor,
        targets: torch.Tensor,
        mtp_pairs: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute cross-entropy losses for the main head and all MTP heads.

        Args:
            main_logits: (bsz, seq, vocab)
            targets:     (bsz, seq) — token IDs; positions with value -100
                         are ignored (padding / prompt masking).
            mtp_pairs:   list of (logits, tgt) pairs as returned by forward().
                         Pass None or [] to compute the main loss only.

        Returns:
            total_loss:  main_loss + weight * mtp_loss
            main_loss:   cross-entropy on main_logits vs targets
            mtp_loss:    mean cross-entropy across MTP depths (0.0 if no pairs)
        """
        main_loss = F.cross_entropy(
            main_logits.reshape(-1, main_logits.size(-1)),
            targets.reshape(-1),
            ignore_index=-100)

        if not mtp_pairs:
            zero = main_loss.new_zeros(())
            return main_loss, main_loss, zero

        # Accumulate from the first valid loss term to keep the gradient graph intact. Using a Python float
        # accumulator + tensor at the end is wrong because it breaks the autograd graph on the first iteration.
        depth_losses: List[torch.Tensor] = []
        for logits, tgt in mtp_pairs:
            # logits: (bsz, usable, vocab), tgt: (bsz, usable)
            # Both are already length-aligned by forward() — no trimming needed.
            if tgt.numel() == 0:
                continue
            depth_losses.append(
                F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    tgt.reshape(-1),
                    ignore_index=-100))

        if depth_losses:
            mtp_loss = torch.stack(depth_losses).mean()
        else:
            mtp_loss = main_loss.new_zeros(())

        total_loss = main_loss + self.mtp_weight * mtp_loss
        return total_loss, main_loss, mtp_loss
