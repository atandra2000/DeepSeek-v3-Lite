import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .mla import MultiHeadLatentAttention
from .moe import DeepSeekMoE


class SwiGLUFFN(nn.Module):
    """SwiGLU FFN: W2(silu(W1(x)) * W3(x))."""
    def __init__(self, dim: int, inter_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim, bias=False)
        self.w2 = nn.Linear(inter_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, inter_dim, bias=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    """Pre-norm MLA + pre-norm SwiGLU/MoE FFN. First `n_dense_layers` are dense."""
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
        x = x + self.attn(self.attn_norm(x), start_pos, mask, use_cache)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class ParallelEmbedding(nn.Module):
    """Vocabulary embedding for single-GPU."""
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.weight = nn.Parameter(torch.empty(vocab_size, dim))
        nn.init.normal_(self.weight, std=0.006)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.embedding(x, self.weight)


class Transformer(nn.Module):
    """DeepSeek-V3-style Transformer: MLA attention, MoE FFN, optional MTP.
    Takes config dict (flat or nested {"model":{...}}). Caches causal mask by seqlen+device.
    Call reset_cache() between independent generation requests."""
    def __init__(self, config: dict, use_checkpoint: bool = False):
        super().__init__()
        model_cfg = config.get("model", config)
        self.use_checkpoint = use_checkpoint
        self.max_seq_len = model_cfg["max_seq_len"]
        self.config = model_cfg
        self.embed = ParallelEmbedding(model_cfg["vocab_size"], model_cfg["dim"])
        self.layers = nn.ModuleList([TransformerBlock(i, model_cfg) for i in range(model_cfg["n_layers"])])
        self.norm = nn.RMSNorm(model_cfg["dim"], eps=1e-6)
        self.weight_tying = model_cfg.get("weight_tying", False)
        self.head = nn.Linear(model_cfg["dim"], model_cfg["vocab_size"], bias=False)
        if self.weight_tying:
            self.head.weight = self.embed.weight
        self._mask_cache: Optional[torch.Tensor] = None
        self._mask_seqlen: int = 0

    def _build_causal_mask(self, seqlen: int, device: torch.device) -> torch.Tensor:
        """Additive causal mask (1,1,S,S). Cached by seqlen+device."""
        if self._mask_cache is None or seqlen != self._mask_seqlen or self._mask_cache.device != device:
            mask = torch.triu(torch.full((seqlen, seqlen), float("-inf"), device=device), diagonal=1)
            self._mask_cache = mask.unsqueeze(0).unsqueeze(0)
            self._mask_seqlen = seqlen
        return self._mask_cache

    def _run_layers(self, h: torch.Tensor, start_pos: int, mask: Optional[torch.Tensor], use_cache: bool) -> torch.Tensor:
        for layer in self.layers:
            if self.use_checkpoint and self.training:
                def _block(h, layer=layer, sp=start_pos, m=mask, uc=use_cache):
                    return layer(h, sp, m, uc)
                h = torch.utils.checkpoint.checkpoint(_block, h, use_reentrant=False)
            else:
                h = layer(h, start_pos, mask, use_cache)
        return h

    def reset_cache(self) -> None:
        for layer in self.layers:
            if hasattr(layer.attn, "reset_cache"):
                layer.attn.reset_cache()

    def moe_layers(self):
        for layer in self.layers:
            if isinstance(layer.ffn, DeepSeekMoE):
                yield layer.ffn

    def forward(self, tokens: torch.Tensor, start_pos: int = 0, use_cache: bool = True) -> torch.Tensor:
        """(bsz, seqlen) -> (bsz, seqlen, vocab_size). start_pos: KV-cache offset."""
        bsz, seqlen = tokens.shape
        h = self.embed(tokens)
        mask = self._build_causal_mask(seqlen, tokens.device) if seqlen > 1 else None
        h = self._run_layers(h, start_pos, mask, use_cache)
        return self.head(self.norm(h))

    def forward_with_hidden(self, tokens: torch.Tensor, start_pos: int = 0, use_cache: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (logits, h_norm). Used by MultiTokenPrediction."""
        bsz, seqlen = tokens.shape
        h = self.embed(tokens)
        mask = self._build_causal_mask(seqlen, tokens.device) if seqlen > 1 else None
        h = self._run_layers(h, start_pos, mask, use_cache)
        h_norm = self.norm(h)
        return self.head(h_norm), h_norm

    @torch.inference_mode()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 512, temperature: float = 1.0,
                 top_p: float = 0.9, top_k: int = 0, eos_token_id: Optional[int] = None) -> torch.Tensor:
        """Autoregressive generation with KV-cache, top-p and top-k sampling.
        Prefill (full prompt) then decode one token at a time."""
        if temperature < 0.0:
            raise ValueError(f"temperature must be >= 0, got {temperature}")
        was_training = self.training
        self.reset_cache()
        self.eval()
        bsz, prompt_len = input_ids.shape
        output = input_ids.clone()
        prefill_logits = self.forward(output, start_pos=0, use_cache=True)
        next_logits = prefill_logits[:, -1, :]
        for step in range(max_new_tokens):
            next_token = self._sample(next_logits, temperature, top_p, top_k)
            output = torch.cat([output, next_token], dim=1)
            if eos_token_id is not None and (next_token == eos_token_id).any():
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
        """Temperature + top-k + top-p sampling. Temperature==0 -> argmax."""
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
    """(total, trainable) — deduplicated by tensor id (shared weights counted once)."""
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
