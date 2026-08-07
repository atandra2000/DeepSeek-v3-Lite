import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .mla import MultiHeadLatentAttention
from .moe import DeepSeekMoE
from ._triton_dispatch import enforce_triton_env_var


class SwiGLUFFN(nn.Module):
    """Feed-forward block using the gated SwiGLU activation."""
    def __init__(self, dim: int, inter_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim, bias=False)
        self.w2 = nn.Linear(inter_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, inter_dim, bias=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the gated feed-forward transformation."""
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    """Pre-norm MLA block with dense or MoE feed-forward layers."""
    def __init__(self, layer_id: int, config: dict):
        super().__init__()
        self.layer_id = layer_id
        self.dim = config["dim"]
        self.n_dense_layers = config["n_dense_layers"]
        self.attn_norm = nn.RMSNorm(self.dim, eps=1e-6)
        self.attn = MultiHeadLatentAttention(config, layer_id)
        self.ffn_norm = nn.RMSNorm(self.dim, eps=1e-6)
        self.ffn = SwiGLUFFN(self.dim, config["inter_dim"]) if layer_id < self.n_dense_layers else DeepSeekMoE(config)

    def forward(self, x: torch.Tensor, start_pos: int = 0, mask: Optional[torch.Tensor] = None, use_cache: bool = True) -> torch.Tensor:
        """Apply attention and feed-forward residual blocks."""
        x = x + self.attn(self.attn_norm(x), start_pos, mask, use_cache)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class Transformer(nn.Module):
    """DeepSeek-V3-style Transformer with MLA and configurable FFN routing."""
    def __init__(self, config: dict, use_checkpoint: bool = False):
        super().__init__()
        model_cfg = config.get("model", config)
        # Enforce the backend policy even when callers construct the model directly.
        enforce_triton_env_var(model_cfg, print)
        self.use_checkpoint = use_checkpoint
        self.max_seq_len = model_cfg["max_seq_len"]
        self.config = model_cfg
        self.embed = nn.Embedding(model_cfg["vocab_size"], model_cfg["dim"])
        nn.init.normal_(self.embed.weight, std=0.006)
        self.layers = nn.ModuleList([TransformerBlock(i, model_cfg) for i in range(model_cfg["n_layers"])])
        self.norm = nn.RMSNorm(model_cfg["dim"], eps=1e-6)
        self.weight_tying = model_cfg.get("weight_tying", False)
        self.head = nn.Linear(model_cfg["dim"], model_cfg["vocab_size"], bias=False)
        if self.weight_tying:
            self.head.weight = self.embed.weight
        self._mask_cache: Optional[torch.Tensor] = None
        self._mask_key: Optional[Tuple[int, int, int, torch.device]] = None

    def _build_causal_mask(self, seqlen: int, kv_len: int, start_pos: int, device: torch.device) -> torch.Tensor:
        """Additive causal mask (1,1,S_q,S_kv), causal by global position.

        `kv_len` is the number of attended keys (`end_pos` when a KV cache
        spans the past, `seqlen` otherwise); `start_pos` offsets the query
        positions so a cached mid-sequence prefill cannot attend its own
        future. Cached by (seqlen, kv_len, start_pos, device).
        """
        key = (seqlen, kv_len, start_pos, device)
        if self._mask_cache is None or key != self._mask_key:
            q = torch.arange(seqlen, device=device)[:, None] + start_pos
            k = torch.arange(kv_len, device=device)[None, :]
            mask = torch.where(q >= k, torch.zeros((), device=device), torch.full((), float("-inf"), device=device))
            self._mask_cache = mask.unsqueeze(0).unsqueeze(0)
            self._mask_key = key
        return self._mask_cache

    def _run_layers(self, h: torch.Tensor, start_pos: int, mask: Optional[torch.Tensor], use_cache: bool) -> torch.Tensor:
        for layer in self.layers:
            if self.use_checkpoint and self.training:
                h = torch.utils.checkpoint.checkpoint(
                    layer, h, start_pos, mask, use_cache, use_reentrant=False,
                )
            else:
                h = layer(h, start_pos, mask, use_cache)
        return h

    def reset_cache(self) -> None:
        """Clear attention caches in every transformer block."""
        for layer in self.layers:
            if hasattr(layer.attn, "reset_cache"):
                layer.attn.reset_cache()

    def moe_layers(self):
        """Yield the MoE layers for diagnostics and bias updates."""
        for layer in self.layers:
            if isinstance(layer.ffn, DeepSeekMoE):
                yield layer.ffn

    def forward(self, tokens: torch.Tensor, start_pos: int = 0, use_cache: bool = True) -> torch.Tensor:
        """Return logits with shape ``(batch, sequence, vocab)``.

        ``start_pos`` is the offset used by the KV cache; integer token buffers
        are normalized to ``torch.long`` at the embedding boundary.
        """
        if tokens.dtype != torch.long:
            tokens = tokens.to(torch.long)
        bsz, seqlen = tokens.shape
        h = self.embed(tokens)
        end_pos = start_pos + seqlen
        if seqlen > 1:
            kv_len = end_pos if use_cache else seqlen
            mask = self._build_causal_mask(seqlen, kv_len, start_pos if use_cache else 0, tokens.device)
        else:
            mask = None
        h = self._run_layers(h, start_pos, mask, use_cache)
        return self.head(self.norm(h))

    def forward_with_hidden(self, tokens: torch.Tensor, start_pos: int = 0, use_cache: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return logits and the pre-final-norm hidden state used by MTP."""
        if tokens.dtype != torch.long:
            tokens = tokens.to(torch.long)
        bsz, seqlen = tokens.shape
        h = self.embed(tokens)
        end_pos = start_pos + seqlen
        if seqlen > 1:
            kv_len = end_pos if use_cache else seqlen
            mask = self._build_causal_mask(seqlen, kv_len, start_pos if use_cache else 0, tokens.device)
        else:
            mask = None
        h = self._run_layers(h, start_pos, mask, use_cache)
        return self.head(self.norm(h)), h

    @torch.inference_mode()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 512, temperature: float = 1.0,
                 top_p: float = 0.9, top_k: int = 0, eos_token_id: Optional[int] = None) -> torch.Tensor:
        """Generate tokens autoregressively with cached attention and sampling."""
        if temperature < 0.0:
            raise ValueError(f"temperature must be >= 0, got {temperature}")
        was_training = self.training
        self.reset_cache()
        self.eval()
        bsz, prompt_len = input_ids.shape
        output = input_ids.clone()
        prefill_logits = self.forward(output, start_pos=0, use_cache=True)
        next_logits = prefill_logits[:, -1, :]
        finished = torch.zeros(bsz, dtype=torch.bool, device=input_ids.device)
        for step in range(max_new_tokens):
            next_token = self._sample(next_logits, temperature, top_p, top_k)
            output = torch.cat([output, next_token], dim=1)
            if eos_token_id is not None:
                finished = finished | (next_token.squeeze(-1) == eos_token_id)
                if finished.all():
                    break
            if output.size(1) >= self.max_seq_len:
                break
            decode_logits = self.forward(next_token, start_pos=prompt_len + step, use_cache=True)
            next_logits = decode_logits[:, -1, :]
        if was_training:
            self.train()
        return output

    @staticmethod
    def _sample(logits: torch.Tensor, temperature: float, top_p: float, top_k: int) -> torch.Tensor:
        """Sample with temperature, top-k, and top-p filtering."""
        if temperature == 0.0:
            return logits.argmax(dim=-1, keepdim=True)
        logits = logits / temperature
        if top_k > 0:
            kth_vals = logits.topk(min(top_k, logits.size(-1)), dim=-1)[0][:, -1:]
            logits = logits.masked_fill(logits < kth_vals, float("-inf"))
        probs = torch.softmax(logits, dim=-1)
        if top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
            cumulative = sorted_probs.cumsum(dim=-1)
            remove = (cumulative - sorted_probs) > top_p
            sorted_probs = sorted_probs.masked_fill(remove, 0.0)
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp(min=1e-10)
            next_token = sorted_idx.gather(-1, torch.multinomial(sorted_probs, num_samples=1))
        else:
            next_token = torch.multinomial(probs, num_samples=1)
        return next_token


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Return deduplicated ``(total, trainable)`` parameter counts."""
    seen = set()
    total = 0
    trainable = 0
    for p in model.parameters():
        pid = id(p)
        if pid in seen:
            continue
        seen.add(pid)
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    return total, trainable
