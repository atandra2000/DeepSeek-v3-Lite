# 07 — DualPipe Bidirectional Pipeline Parallelism

> **Canonical** for DeepSeek-V3's DualPipe algorithm: the pipeline-bubble problem, 1F1B scheduling, and the bidirectional overlap that halves the bubble. Educational textbook chapter — from-scratch, with the schedule diagrams and the intuition.

> How DeepSeek-V3 overlaps forward/backward GEMMs with All-to-All MoE dispatch so distributed MoE training spends GPU time computing instead of waiting on the network. **Status in this repo:** DualPipe is **paper-spec only — not implemented.** This reproduction is single-GPU (`training/pretrain.py` is an explicit single-device loop, no `torch.distributed`, no pipeline stages). This chapter documents the technique because it is central to DeepSeek-V3's distributed story; at 422 M params on one A100, there is nothing to pipeline across.

**Depends on:** [[Docs/04_DeepSeekMoE]], [[Docs/06_FP8_Mixed_Precision]] · **Read next:** [[Docs/08_Training_Pipeline]]

---

## 0. Status in this repo

| Aspect | DeepSeek-V3 (paper) | DeepSeek-v3-Lite (this repo) |
|---|---|---|
| Device count | many GPUs across nodes | **1 GPU** |
| Parallelism | DualPipe (pipeline) + expert/data parallel | none — single device |
| All-to-All token dispatch | yes (MoE routing crosses nodes) | no — experts live on one device |
| Pipeline stages | model split across GPUs | no split |
| Bubble ratio | halved by bidirectional overlap | n/a |

`grep` for `dualpipe`, `dual_pipe`, `pipeline.*parallel`, `all_to_all` in the repo returns nothing — these are not in the code path. The rest of this chapter explains the V3 algorithm.

---

## 1. The problem: pipeline bubbles from sequential dependencies

Pipeline parallelism splits a model's layers across `P` GPUs (stages). Micro-batch `b` cannot enter stage `k` until stage `k-1` has produced its output for `b`, and the backward pass has the mirror dependency. The classic **GPipe** schedule fills the pipeline, then drains it:

```
Stage 0: F1 F2 F3 F4 .  .  .  .  B4 B3 B2 B1
Stage 1: .  F1 F2 F3 F4 .  .  B4 B3 B2 B1 .
Stage 2: .  .  F1 F2 F3 F4 .  B4 B3 B2 B1 .  .
Stage 3: .  .  .  F1 F2 F3 F4 B4 B3 B2 B1 .  .
         ↑fill (idle)↑       ↑drain (idle)↑
```

The "fill" and "drain" regions are **bubbles** — stages idle because their dependency isn't ready. For `P` stages and `M` micro-batches, the bubble fraction is roughly `(P-1)/M` of the ideal compute time. At `P=8` stages and `M=8` micro-batches that's ~⅞ of one micro-batch wasted per stage — tolerable but not free, and it grows with `P`.

---

## 2. 1F1B — the first improvement

**1F1B (one forward, one backward)** interleaves forward and backward passes to cap the activation memory and overlap fill with drain. Instead of doing all forwards then all backwards, each stage does one forward as soon as possible, then a backward, in a steady state:

```
Stage 0: F1 F2 F3 F4 B1 B2 B3 B4
Stage 1: .  F1 F2 F3 B1 F4 B2 B3 B4
...
```

1F1B shrinks the *memory* footprint (each stage holds at most `P` activations) and tightens the bubble, but the bubble is still `(P-1)/M` and it is **blocking**: while a stage waits for its dependency, it does nothing. DualPipe attacks the *blocking* part.

---

## 3. Why MoE makes the bubble worse — All-to-All

In a dense pipeline, the inter-stage dependency is a point-to-point activation send. In **MoE**, a token routed to expert `j` living on a *different* node must travel by an **All-to-All** collective — every GPU exchanges tokens with every other GPU. All-to-All is a heavy, latency-bound collective (the network is the bottleneck, not bandwidth). In a naive schedule, the GPU **stalls** during the All-to-All: it has dispatched its tokens and is waiting for the incoming tokens before it can compute the next layer.

```
GPU i:  [compute layer k] ──> [All-to-All dispatch] ──> [IDLE: waiting for incoming] ──> [compute layer k+1]
```

That idle window is the prize. DualPipe's whole idea: **the GPU has a second stream of work it could be doing during that wait.**

---

## 4. DualPipe — two streams in opposite directions

DualPipe runs **two interleaved pipeline streams in opposite directions** over the same stages:

```
Forward stream A  (stage 0 → 1 → 2 → 3):   F0 → F1 → F2 → F3
Forward stream B  (stage 3 → 2 → 1 → 0):   F3 ← F2 ← F1 ← F0
```

Each stage holds two chunks of work: one from stream A (moving left→right) and one from stream B (moving right→left). The schedule is built so that **when stage `i` is doing the All-to-All dispatch for stream A's layer `k`, its Tensor Cores are computing stream B's layer `k-1`** (and vice versa). The All-to-All communication is **fully hidden behind** compute from the opposing stream.

```
Stage i timeline:
  [compute B: layer k-1] ──┐  [compute A: layer k]   ──┐
  [All-to-All A: layer k] ─┘  [All-to-All B: layer k] ─┘
       ↑ overlapped ↑            ↑ overlapped ↑
```

`★ Insight ─────────────────────────────────────`
The deep idea is **bidirectional symmetry as a scheduling resource.** A one-directional pipeline has nothing to do during a communication stall — the work all flows the same way, so a wait is a wait. A *bidirectional* pipeline always has a counter-stream whose compute is independent of the stalled collective. DualPipe pairs each communication with the opposing stream's compute by construction; the bubble is not eliminated (the dependency still exists) but it is *hidden* because the GPU is never asked to be idle while it has an independent task queued. This is the same principle as async I/O overlap, applied to the pipeline graph.
`─────────────────────────────────────────────────`

---

## 5. The bubble math

| Schedule | Bubble ratio (P stages, M micro-batches) |
|---|---|
| GPipe (fill-drain) | `(P-1)/M` |
| 1F1B | `(P-1)/M` (blocking) |
| **DualPipe** | **`(P-1)/(2M)` — halved** |

DualPipe halves the bubble because the two streams share the fill/drain cost: the forward-fill of stream A overlaps the backward-drain of stream B and vice versa. The asymptotic improvement is 2×, and the constant factor (communication fully hidden behind compute) is the larger practical win.

---

## 6. Why DualPipe is hard — the engineering surface

DualPipe is not a flag you flip; it requires:

1. **Model chunking** — each stage holds two halves of its layers (one per stream), so memory layout and the forward/backward graph must support dual streams without aliasing.
2. **A hand-built schedule** — the exact ordering of which stream computes which layer when, per stage, is a constraint-satisfaction problem (dependencies + communication + compute durations). DeepSeek-V3's schedule is hardcoded, not auto-derived.
3. **Forward/backward recomputation** — to fit both streams' activations, DualPipe recomputes the forward in the backward (gradient checkpointing per stream), so the memory for two streams stays near one stream's worth.
4. **Collective tuning** — All-to-All must be sized to fit inside the compute window of the opposing stream's layer; if comms exceed compute, the bubble returns. This couples kernel choice, MoE expert count, and network topology.

This is the engineering surface that makes DualPipe a *distributed-systems* project, not a model-architecture one. It is orthogonal to MLA/MoE/MTP — you could train the same model with plain 1F1B and lose the 2× bubble, or with DualPipe and gain it, on the *same* architecture.

---

## 7. Why this repo skips it — the single-GPU scope

DualPipe's value is realized only when (a) the model is split across ≥2 GPUs and (b) MoE All-to-All is a measurable stall. This repo's `training/pretrain.py` is an explicit **single-GPU** loop: one device, `torch.compile(max-autotune)`, gradient checkpointing, no `torch.distributed` process group, no `all_to_all`. The MoE experts all live on the one device, so routing is a `stacked`/`triton_grouped` local dispatch (see [[Docs/12_Triton_Kernels]]), not a cross-node collective. There is nothing to pipeline.

The choice is deliberate and documented in the design goals (see [[Docs/02_Model_Architecture]] §Design Goals): *single-GPU training* and *one file explains the full train path* are both portfolio-defensible properties, and both are incompatible with pipeline parallelism. DualPipe is documented here so the portfolio covers the full DeepSeek-V3 distributed story; it is the natural next chapter if the model were scaled to a size that no longer fit on one device.

> **Next:** [[Docs/08_Training_Pipeline]] — the actual single-GPU pretrain loop, AdamW, μP LR scaling, NaN guard, and atomic checkpointing.