# training/rl.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Callable
import copy
from torch.amp import autocast


# ── Config ─────────────────────────────────────────────────────────────────────

@dataclass
class GRPOConfig:
    batch_size: int = 4
    group_size: int = 4          # samples per prompt (single-GPU cap)
    epsilon: float = 0.2         # PPO clip ratio
    beta: float = 0.04           # KL penalty coefficient
    lr: float = 1e-6
    max_steps: int = 500
    max_grad_norm: float = 1.0
    temperature: float = 0.7
    max_new_tokens: int = 512


# ── Rule-based reward (the only reward source on a single A100) ────────────────

def rule_based_reward(response: str, question: str = "") -> float:
    """
    Heuristic reward for verifiable reasoning tasks.
    """
    score = 0.0
    if "\\boxed{" in response:
        score += 0.5
    if any(w in response.lower() for w in ("therefore", "step", "because", "∴")):
        score += 0.3
    if len(response.split()) > 20:
        score += 0.2
    return min(score, 1.0)


# ── GRPO Trainer ───────────────────────────────────────────────────────────────

class GRPOTrainer:
    """
    Group Relative Policy Optimisation (GRPO) for a single A100 80GB.
    """

    def __init__(
        self,
        policy_model: nn.Module,
        config: GRPOConfig,
    ):
        self.policy = policy_model
        self.config = config

        self.reference = copy.deepcopy(policy_model)
        for p in self.reference.parameters():
            p.requires_grad_(False)
        self.reference.eval()

        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(), lr=config.lr, weight_decay=0.0
        )

    @torch.no_grad()
    def _compute_log_probs(
        self, model: nn.Module, token_ids: torch.Tensor
    ) -> torch.Tensor:
        logits = model(token_ids[:, :-1])
        log_p = F.log_softmax(logits, dim=-1)
        return log_p.gather(-1, token_ids[:, 1:].unsqueeze(-1)).squeeze(-1)

    def compute_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        mean = rewards.mean()
        std = rewards.std().clamp(min=1e-8)
        return (rewards - mean) / std

    def train_step(
        self,
        prompts: List[str],
        generate_fn: Callable,
        tokenizer,
    ) -> dict:
        self.policy.train()
        self.optimizer.zero_grad(set_to_none=True)
        device = next(self.policy.parameters()).device

        total_policy_loss = 0.0
        total_kl_loss = 0.0
        n_updates = 0

        for prompt in prompts:
            responses = [generate_fn(prompt, self.policy) for _ in range(self.config.group_size)]

            rewards = torch.tensor(
                [rule_based_reward(r, prompt) for r in responses],
                dtype=torch.float32,
                device=device,
            )

            advantages = self.compute_advantages(rewards)

            prompt_loss = torch.tensor(0.0, device=device)

            for i, response in enumerate(responses):
                if isinstance(response, str):
                    ids = tokenizer.encode(response, return_tensors="pt").to(device)
                else:
                    ids = response.unsqueeze(0).to(device) if response.dim() == 1 else response.to(device)

                if ids.numel() < 2:
                    continue

                with torch.enable_grad(), autocast("cuda", dtype=torch.bfloat16):
                    logits = self.policy(ids[:, :-1])
                    log_p = F.log_softmax(logits, dim=-1)
                    token_log_probs = log_p.gather(
                        -1, ids[:, 1:].unsqueeze(-1)
                    ).squeeze(-1)

                with torch.no_grad(), autocast("cuda", dtype=torch.bfloat16):
                    ref_logits = self.reference(ids[:, :-1])
                    ref_log_p = F.log_softmax(ref_logits, dim=-1)
                    ref_token_log_probs = ref_log_p.gather(
                        -1, ids[:, 1:].unsqueeze(-1)
                    ).squeeze(-1)

                seq_log_ratio = (token_log_probs - ref_token_log_probs).sum(-1)
                # Clamp log-ratio so exp() does not overflow with sharp divergence.
                seq_log_ratio = seq_log_ratio.clamp(-20.0, 20.0)
                ratio = seq_log_ratio.exp()

                adv = advantages[i]
                clipped = torch.clamp(ratio, 1.0 - self.config.epsilon, 1.0 + self.config.epsilon)
                policy_loss = -torch.min(ratio * adv, clipped * adv).mean()

                kl = (token_log_probs.exp() * (token_log_probs - ref_token_log_probs)).sum(-1).mean()
                kl_loss = self.config.beta * kl

                prompt_loss = prompt_loss + (policy_loss + kl_loss) / self.config.group_size
                total_kl_loss += kl_loss.item()
                n_updates += 1

            total_policy_loss += prompt_loss.item()
            prompt_loss.backward()

        nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
        self.optimizer.step()

        n = max(n_updates, 1)
        return {
            "policy_loss": total_policy_loss / len(prompts),
            "kl_loss": total_kl_loss / n,
        }

    def update_reference(self):
        self.reference.load_state_dict(self.policy.state_dict())
