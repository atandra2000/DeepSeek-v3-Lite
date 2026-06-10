# inference/speculative.py
"""
Speculative decoding via Multi-Token Prediction (MTP).

Architecture (DeepSeek-V3, Section 5.4.3):
  1. Main model predicts token T1 from input.
  2. MTP draft head predicts token T2 using T1's embedding + main model hidden state.
  3. Main model verifies T2 at the next step; accept if p_main(T2) / p_draft(T2) > threshold.
"""
import sys
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn

sys.path.append(str(Path(__file__).parent.parent))
from models.mtp import MTPModule


class SpeculativeDecoder:
    """
    MTP-based speculative decoder: one draft token per step.
    Verifies against main model; accepts or falls back.
    """

    def __init__(
        self,
        main_model: nn.Module,
        mtp_module: MTPModule,
        acceptance_threshold: float = 0.8,
    ):
        self.main_model = main_model
        self.mtp = mtp_module
        self.threshold = acceptance_threshold

    @torch.inference_mode()
    def generate_step(
        self,
        input_ids: torch.Tensor,
        start_pos: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        """
        Single speculative step.

        Returns T1 (main model's greedy token) and T2 (draft head's speculative token)
        separately so the caller can always append T1 and optionally also T2.

        Args:
            input_ids: (1, seq) token IDs fed into the main model
            start_pos: KV-cache start position
        """
        # Main model: predict T1 from the current context
        main_logits = self.main_model(input_ids, start_pos=start_pos, use_cache=True)
        if main_logits.dim() == 3:
            main_logits = main_logits[:, -1, :]
        main_probs = torch.softmax(main_logits, dim=-1)
        token_main = main_probs.argmax(dim=-1)

        # Run the MTP head in the *next* depth — we need the last hidden state
        # of the main model, which we get cheaply from the same forward call.
        # For a single-step decoder we approximate: re-run forward_with_hidden
        # once with the freshly-appended main token to obtain a hidden state
        # that includes T1.
        next_input = torch.cat([input_ids, token_main.unsqueeze(0)], dim=1)
        _, hidden = self.main_model.forward_with_hidden(next_input)
        hidden_last = hidden[:, -1:, :]

        token_main_emb = self.main_model.embed(token_main.unsqueeze(-1))
        draft_logits, _ = self.mtp(hidden_last, token_main_emb)
        draft_logits = draft_logits[:, -1, :]
        draft_probs = torch.softmax(draft_logits, dim=-1)
        token_draft = draft_probs.argmax(dim=-1)

        # Acceptance test (probability-ratio heuristic)
        p_main_of_draft  = main_probs[0, token_draft[0]].item()
        p_draft_of_draft = draft_probs[0, token_draft[0]].item()
        acceptance_ratio = (
            min(1.0, p_main_of_draft / p_draft_of_draft)
            if p_draft_of_draft > 1e-12 else 0.0
        )
        was_accepted = acceptance_ratio >= self.threshold

        return token_main, token_draft, was_accepted

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 512,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Full speculative generation loop.

        Each step produces 1 token (main model's prediction) or 2 tokens
        (main + draft) when the draft is accepted.
        """
        output = input_ids.clone()
        n_generated = 0

        # Reset the KV cache on entry to avoid stale context bleed.
        if hasattr(self.main_model, "reset_cache"):
            self.main_model.reset_cache()

        while n_generated < max_new_tokens:
            # start_pos advances by the number of tokens already produced
            start_pos = output.size(1) - 1
            token_main, token_draft, was_accepted = self.generate_step(
                output, start_pos=start_pos
            )

            output = torch.cat([output, token_main.unsqueeze(0)], dim=1)
            n_generated += 1

            if was_accepted and n_generated < max_new_tokens:
                output = torch.cat([output, token_draft.unsqueeze(0)], dim=1)
                n_generated += 1

        return output
