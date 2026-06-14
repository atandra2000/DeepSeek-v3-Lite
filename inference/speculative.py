"""Speculative decoding via Multi-Token Prediction (MTP): main model → draft → accept/reject."""
import sys
from pathlib import Path
from typing import Optional, Tuple
import torch
import torch.nn as nn
sys.path.append(str(Path(__file__).parent.parent))
from models.mtp import MTPModule


class SpeculativeDecoder:
    """MTP-based speculative decoder: main model predicts T1, draft predicts T2; verify and accept or fall back."""

    def __init__(self, main_model: nn.Module, mtp_module: MTPModule, acceptance_threshold: float = 0.8):
        self.main_model = main_model
        self.mtp = mtp_module
        self.threshold = acceptance_threshold

    @torch.inference_mode()
    def generate_step(self, last_token: torch.Tensor, start_pos: int) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        main_logits = self.main_model(last_token, start_pos=start_pos, use_cache=True)
        main_probs = torch.softmax(main_logits[:, -1, :], dim=-1)
        token_main = main_probs.argmax(dim=-1)
        t1_pos = start_pos + 1
        _, hidden = self.main_model.forward_with_hidden(token_main.unsqueeze(0), start_pos=t1_pos, use_cache=True)
        hidden_last = hidden[:, -1:, :]
        token_main_emb = self.main_model.embed(token_main.unsqueeze(-1))
        draft_logits, _ = self.mtp(hidden_last, token_main_emb)
        draft_probs = torch.softmax(draft_logits[:, -1, :], dim=-1)
        token_draft = draft_probs.argmax(dim=-1)
        p_main_of_draft = main_probs[0, token_draft[0]].item()
        p_draft_of_draft = draft_probs[0, token_draft[0]].item()
        acceptance_ratio = min(1.0, p_main_of_draft / p_draft_of_draft) if p_draft_of_draft > 1e-12 else 0.0
        return token_main, token_draft, acceptance_ratio >= self.threshold

    @torch.inference_mode()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 512, temperature: float = 1.0,
                 eos_token_id: Optional[int] = None) -> torch.Tensor:
        output = input_ids.clone()
        n_generated = 0
        if hasattr(self.main_model, "reset_cache"):
            self.main_model.reset_cache()
        _ = self.main_model(output, start_pos=0, use_cache=True)
        while n_generated < max_new_tokens:
            start_pos = output.size(1) - 1
            last_token = output[:, -1:]
            token_main, token_draft, was_accepted = self.generate_step(last_token, start_pos=start_pos)
            output = torch.cat([output, token_main.unsqueeze(0)], dim=1)
            n_generated += 1
            if eos_token_id is not None and token_main.item() == eos_token_id:
                break
            if was_accepted and n_generated < max_new_tokens:
                output = torch.cat([output, token_draft.unsqueeze(0)], dim=1)
                n_generated += 1
                if eos_token_id is not None and token_draft.item() == eos_token_id:
                    break
        return output
