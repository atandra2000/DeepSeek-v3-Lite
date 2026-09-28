# DeepSeek-v3-Lite — Multi-Token Prediction (MTP)

> **Audience:** beginner. The only prerequisite is the causal-LM objective, restated below.
>
> **Read this if** you want the one-page story of the third pillar: why the model predicts *two* tokens per position during training, and how that same head drafts tokens at inference. **Skip if** you want the length-alignment algebra in full and the loss wiring → [DeepSeekMoE & MTP](moe-mtp.md), [R5 — MTP API](../references/R5_mtp_api.md).

**Depends on:** nothing beyond next-token prediction · **Read next:** [DeepSeekMoE & MTP](moe-mtp.md), [Inference & Serving](../inference.md)

---

## 1. Prerequisite: what a causal LM already does

A causal LM reads tokens `t₁ … tₙ`, and at every position `i` predicts `t_{i+1}` from the hidden state `hᵢ`. One position, one prediction, one unit of training signal per token.

## 2. The idea: one extra prediction per position

MTP adds a small **depth-1 head** that answers a second question from the same forward pass: *"given where you are now **and** the token that actually came next, what comes **after** that?"* Concretely, position `i` predicts `t_{i+2}` from the pair `(hᵢ, embed(t_{i+1}))`. Each position now supervises two future tokens instead of one — the training signal is **denser**, and hidden states are pushed to encode a bit of the future, not just the present.

The canonical config runs `mtp_depth: 1` with `mtp_loss_weight: 0.3` (`configs/pretrain_a100_422m.yaml`); total parameters rise from 411.6 M to **418.7 M** — the head adds ~7 M parameters [derived from the README counts].

## 3. The block, and the shared head

`models/mtp.py:MTPBlock` fuses the two inputs — RMSNorm'd main-model hidden state and RMSNorm'd next-token embedding, concatenated and squashed back to `dim` by one linear — then refines the fused sequence with a causal self-attention layer and a SwiGLU FFN, each in a pre-norm residual. Because attention is causal, position `i` only attends to fusions at `≤ i`, so the refinement cannot leak future tokens into the prediction.

The head **shares** the main model's embedding table and LM head: `models/mtp.py:MultiTokenPrediction` attaches `main_model.embed` and passes `main_model.head` to every MTP module via `models/mtp.py:MTPModule.set_output_head`. No second vocab matrix, and the draft distribution lives in exactly the same space as the main distribution — which is what makes speculative decoding cheap.

## 4. Length alignment, stated once

Predicting `t_{i+2}` from `hᵢ` means the last two positions of a sequence have no ground truth for a depth-1 head. `models/mtp.py:MultiTokenPrediction.forward` crops everything to the usable prefix — for depth `d`, `usable = seq_len − d − 2`:

```python
# illustrative — the depth-0 slice arithmetic, models/mtp.py:MultiTokenPrediction.forward
h_in   = prev_h[:, :usable]                       # main-model hidden states at positions 0..usable-1
emb_in = self.embed(tokens[:, d+1 : d+1+usable])  # ground-truth token t_{i+1}
tgt    = tokens[:, d+2 : d+2+usable]              # label t_{i+2}
```

Deeper heads (depth 2+) would chain on the refined hidden states; this repo ships depth 1.

## 5. The loss

`models/mtp.py:MultiTokenPrediction.compute_loss` returns

`total = main_loss + 0.3 · mean(depth losses)`,

with each depth loss an ordinary cross-entropy over its aligned slice. The trainer (`training/pretrain.py:Pretrainer`) trains the wrapper, so the shared embedding/head get gradients from **both** objectives while the MTP block alone owns the auxiliary gradient. When MTP is disabled (`mtp_depth: 0`), the wrapper is never built and the loss is the plain main loss.

## 6. The payoff at inference: speculative decoding

The trained head is a free draft model. `inference/speculative.py:SpeculativeDecoder.generate_step` runs the main model for one token, then feeds that token's embedding + hidden state to the MTP head to **draft the next token in one forward**, and accepts it if the main model's own probability of that draft clears a threshold. Accepted drafts make decode up to ~2 tokens per main-model step [INFERENCE — acceptance rate depends on the checkpoint and threshold; nothing here is measured]. Full procedure: [Inference & Serving](../inference.md) and [R9](../references/R9_inference_api.md).

<!-- docs:verified 2026-09-21 · d9a5de4 -->
