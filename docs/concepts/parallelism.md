# DeepSeek-v3-Lite — DualPipe Parallelism (Paper-Spec)

> **Canonical** for DeepSeek-V3's DualPipe algorithm: the pipeline-bubble problem, 1F1B scheduling, and the bidirectional overlap that halves the bubble. Educational textbook chapter — from-scratch, with the schedule diagrams and the intuition.

> How DeepSeek-V3 overlaps forward/backward GEMMs with All-to-All MoE dispatch so distributed MoE training spends GPU time computing instead of waiting on the network. **Status in this repo:** DualPipe is **paper-spec only — not implemented.** This reproduction is single-GPU (`training/pretrain.py` is an explicit single-device loop, no `torch.distributed`, no pipeline stages). This chapter documents the technique because it is central to DeepSeek-V3's distributed story; at 411.6M params on one A100, there is nothing to pipeline across.

**Depends on:** [MoE & MTP](../concepts/moe-mtp.md), [MLA & Mixed Precision](../concepts/attention-and-precision.md) · **Read next:** [Training](../training.md)

---

## Table of Contents

1. [Status in this repo](#0-status-in-this-repo)
2. [Why pipeline parallelism exists](#1-why-pipeline-parallelism-exists)
3. [Naive schedules: GPipe fill–drain](#2-naive-schedules-gpipe-filldrain)
4. [Micro-batch schedules: 1F1B](#3-micro-batch-schedules-1f1b)
5. [Interleaved schedules](#4-interleaved-schedules)
6. [Why MoE makes the bubble worse — All-to-All](#5-why-moe-makes-the-bubble-worse--all-to-all)
7. [DualPipe — two streams in opposite directions](#6-dualpipe--two-streams-in-opposite-directions)
8. [The bubble math, collected](#7-the-bubble-math-collected)
9. [Why DualPipe is hard — the engineering surface](#8-why-dualpipe-is-hard--the-engineering-surface)
10. [Why this repo skips it — the single-GPU loop](#9-why-this-repo-skips-it--the-single-gpu-loop)
11. [Pitfalls](#10-pitfalls)
12. [Check your understanding](#11-check-your-understanding)

---

## 0. Status in this repo

| Aspect | DeepSeek-V3 (paper) | DeepSeek-v3-Lite (this repo) |
|---|---|---|
| Device count | many GPUs across nodes | **1 GPU** |
| Parallelism | DualPipe (pipeline) + expert/data parallel | none — single device |
| All-to-All token dispatch | yes (MoE routing crosses nodes) | no — experts live on one device |
| Pipeline stages | model split across GPUs | no split |
| Bubble ratio | halved by bidirectional overlap | n/a |

`grep` for `dualpipe`, `dual_pipe`, `pipeline.*parallel`, `all_to_all`, `torch.distributed` in the repo returns nothing except this chapter — these are not in the code path. The rest of this chapter explains the V3 algorithm, then walks the actual single-GPU loop that stands in for it here.

---

## 1. Why pipeline parallelism exists

### 1.1 The single-GPU wall

A transformer's layers are **sequential**: layer $k+1$ cannot start on a token until layer $k$ has finished. One GPU can therefore only overlap *micro-batches* (independent training samples) with each other — and inside one GPU, that overlap is the job of the kernel scheduler, which has a finite window of parallelism. When the model is small enough that a single device holds it (this repo: 411.6M params, ~185M active per token), the wall never arrives. When the model is 671B with 37B active parameters (DeepSeek-V3), no single device can hold the weights, let alone the activations, so the layers must be **split across devices** — and then the sequential dependency becomes a *network* dependency: every token that flows from stage $k$ to stage $k+1$ must cross the interconnect.

### 1.2 The formal model

Define the vocabulary used by every schedule below:

- $P$ — number of **pipeline stages**: contiguous groups of layers, each resident on one device (or one device's share).
- $M$ — number of **micro-batches**: the global training batch is chopped into $M$ independent pieces that flow through the pipeline.
- $F$ — wall-clock time for one stage's forward pass on one micro-batch.
- $B$ — wall-clock time for one stage's backward pass on one micro-batch. For transformers $B \approx 2F$ in FLOPs, but for the *scheduling* analysis only the sum $F + B$ matters.
- **Bubble** — a stage is idle because its input dependency (upstream forward, or downstream gradient) has not arrived, while other stages still have work.

The two structural constraints:

1. **Forward chain:** micro-batch $m$ enters stage $k$ only after stage $k-1$ emitted it.
2. **Backward chain:** stage $k$'s backward for micro-batch $m$ needs the activation-gradients from stage $k+1$'s backward for the *same* $m$, plus its own saved forward activations.

A schedule is an assignment of (stage, micro-batch, fwd/bwd) to time slots that respects both chains. The bubble fraction is the fraction of each stage's wall time that is idle; the goal of every schedule in this chapter is to shrink it.

---

## 2. Naive schedules: GPipe fill–drain

The classic **GPipe** schedule (Huang et al., 2019) runs all $M$ forwards through every stage, then all $M$ backwards. The pipeline fills, then drains:

```
Stage 0: F1 F2 F3 F4 .  .  .  .  B4 B3 B2 B1
Stage 1: .  F1 F2 F3 F4 .  .  B4 B3 B2 B1 .
Stage 2: .  .  F1 F2 F3 F4 .  B4 B3 B2 B1 .  .
Stage 3: .  .  .  F1 F2 F3 F4 B4 B3 B2 B1 .  .
         ↑fill (idle)↑       ↑drain (idle)↑
```

The "fill" region (stages 1–3 waiting for stage 0's first forwards) and the "drain" region (stage 0 waiting for the backward wave to come back) are **bubbles**. Derive the exact cost. Each stage is busy for $M(F+B)$. The last operation in the whole run is stage $P-1$'s last backward; the first forward reaches stage $P-1$ after $(P-1)F$, the $M$ forwards then $M$ backwards flow through it, so

$$T_{\text{GPipe}} = (M + P - 1)(F + B).$$

Per-stage idle time is $T_{\text{GPipe}} - M(F+B) = (P-1)(F+B)$, so the bubble fraction is

$$\frac{(P-1)(F+B)}{(M+P-1)(F+B)} = \frac{P-1}{M+P-1} \;\xrightarrow[M \gg P]{}\; \frac{P-1}{M}.$$

Two observations worth locking in:

- The $F/B$ ratio cancels — the bubble fraction depends only on $P$ and $M$. (The *absolute* idle time $(P-1)(F+B)$ does depend on the ratio: since $B \approx 2F$, a real run's drain is about twice as long as its fill.)
- The asymptotic form $(P-1)/M$ is what every pipeline paper quotes; it is exact only for $M \gg P$. At $P=4, M=4$ the exact value is $3/7 \approx 43\%$, while the asymptotic form says $75\%$ — use the exact form for small $M$.

Sanity check with the diagram above ($P=4, M=4, F=B=1$): $T = 7 \cdot 2 = 14$ units, idle per stage $= 6$ units, fraction $6/14 = 3/7$ ✓.

---

## 3. Micro-batch schedules: 1F1B

**1F1B (one forward, one backward)** — from PipeDream (Narayanan et al., 2021) — interleaves forward and backward passes at each stage. Once the pipeline is warm, a stage alternates: forward when its input is ready, backward when its gradient is ready, instead of doing all forwards first. The steady-state *shape* is:

```
# illustrative — steady-state alternation per stage (P=4, M=4, F=B=1)
Stage 0: F1 F2 F3 F4 .  B1 B2 B3 B4
Stage 1: .  F1 F2 F3 B1 F4 B2 B3 B4
Stage 2: .  .  F1 F2 B1 F3 B2 F4 B3 B4
Stage 3: .  .  .  F1 B1 F2 B2 F3 B3 F4 B4
```

The key property is **memory, not time**. In GPipe, a stage must hold the activations of *all* $M$ micro-batches it has forwarded but not yet backwarded — $O(M)$ activations per stage. In 1F1B, each stage holds at most $P$ micro-batch activation sets (a stage can only be ahead of its downstream neighbor by the pipeline depth), i.e. $O(P)$. With $P$ fixed and $M$ chosen large enough to amortize the bubble, the memory win is what matters:

- Same asymptotic bubble as GPipe: $(P-1)/M$ (both schedules are throughput-equivalent at fixed $M$ — 1F1B does not shrink the bubble *for a given $M$*, it shrinks the *memory*).
- Because activation memory no longer grows with $M$, the practitioner can raise $M$ at fixed memory until the bubble $(P-1)/M$ is negligible. GPipe is memory-bound before it is bubble-bound; 1F1B is only bubble-bound.

In short: **1F1B converts the $O(M)$ memory tax of GPipe into the ability to run many more micro-batches, which is what actually drives the bubble down.**

---

## 4. Interleaved schedules

Megatron-LM's **interleaved** schedule is the next refinement: instead of giving each stage $L/P$ consecutive layers, the $L$ layers are split into $v$ **chunks** and chunks are assigned to stages in round-robin. Each stage holds $v$ disjoint layer-chunks (e.g. layers $0, P, 2P, \dots$ rather than layers $0..L/P-1$). Every micro-batch now visits each stage $v$ times, and the fill/drain per visit costs only $1/v$ of a full stage pass:

- Per-chunk forward time $F_v = F/v$, backward $B_v = B/v$.
- Fill+drain per stage = $(P-1)(F_v + B_v) = (P-1)(F+B)/v$.
- Total busy work is unchanged, so

$$\text{bubble}_{\text{interleaved}} \approx \frac{P-1}{vM}.$$

The bubble shrinks by the interleave factor $v$ — for $P=8, M=40, v=2$: $7/80 \approx 8.75\%$ vs $17.5\%$ non-interleaved. The cost: each chunk boundary is another activation transfer between stages, so communication volume grows linearly in $v$ (a token's activations cross the interconnect $2v$ times per forward+backward instead of 2). Interleaving is a *bubble-vs-bandwidth* trade, tuned by $v$.

DualPipe is best understood as the next step in this same lineage: it halves the bubble *without* increasing the communication volume, by exploiting symmetry instead of finer chunking.

---

## 5. Why MoE makes the bubble worse — All-to-All

In a dense pipeline, the inter-stage dependency is a point-to-point activation send. In **MoE**, the FFN of each layer routes every token to its top-$k$ experts, and when experts are spread across devices, the token *payloads* must travel: **All-to-All**. Every device simultaneously exchanges token tensors with every other device — the collective where all $D$ devices send to all $D$ devices. All-to-All is **latency-bound**: it is a rendezvous of $D^2$ transfers that all must complete before any stage can start its expert GEMMs. The network, not the Tensor Cores, is the critical resource:

```
GPU i:  [compute layer k] ──> [All-to-All dispatch] ──> [IDLE: waiting for incoming] ──> [compute layer k+1]
```

That idle window is the prize. There are two distinct ways to attack it, and DualPipe uses both:

**Intra-stage overlap (both in V3 and in any well-tuned MoE pipeline).** The all-to-all that dispatches the FFN input of layer $k+1$ does not depend on the attention compute of layer $k+1$ — only on layer $k$'s output. So a stage can issue the dispatch for layer $k+1$ *before* starting layer $k+1$'s attention GEMMs, and the network transfer runs concurrently with Tensor-Core work. This hides communication behind *adjacent* compute within the same stream:

```
GPU i, layer k+1:  [dispatch FFN tokens] ─┐
                  [attention GEMMs]      ─┘  both in flight
```

**Cross-stream overlap (the DualPipe-specific half).** Intra-stage overlap only helps if there is compute to run while the collective is in flight. In a one-directional pipeline, when stage $i$ has dispatched its tokens and is waiting, there is *no* independent compute available — the work all flows one way, so a wait is a wait. This is the gap DualPipe fills.

---

## 6. DualPipe — two streams in opposite directions

DualPipe runs **two interleaved pipeline streams in opposite directions** over the same stages:

```
Forward stream A  (stage 0 → 1 → 2 → 3):   F0 → F1 → F2 → F3
Forward stream B  (stage 3 → 2 → 1 → 0):   F3 ← F2 ← F1 ← F0
```

Each stage holds two chunks of work: one from stream A (moving left→right) and one from stream B (moving right→left). The schedule is built so that **when stage $i$ is doing the All-to-All dispatch for stream A's layer $k$, its Tensor Cores are computing stream B's layer $k-1$** (and vice versa). The All-to-All communication is **fully hidden behind** compute from the opposing stream:

```
Stage i timeline:
  [compute B: layer k-1] ──┐  [compute A: layer k]   ──┐
  [All-to-All A: layer k] ─┘  [All-to-All B: layer k] ─┘
       ↑ overlapped ↑            ↑ overlapped ↑
```

`★ Insight ─────────────────────────────────────`
The deep idea is **bidirectional symmetry as a scheduling resource.** A one-directional pipeline has nothing to do during a communication stall — the work all flows the same way, so a wait is a wait. A *bidirectional* pipeline always has a counter-stream whose compute is independent of the stalled collective. DualPipe pairs each communication with the opposing stream's compute by construction; the bubble is not eliminated (the dependency still exists) but it is *hidden* because the GPU is never asked to be idle while it has an independent task queued. This is the same principle as async I/O overlap, applied to the pipeline graph.
`─────────────────────────────────────────────────`

Because the two streams move in opposite directions, their fill/drain phases are anti-correlated: while stream A is still *filling* (upstream stages waiting), stream B is already *draining* (downstream stages finishing) — one stream's bubble coincides with the other stream's busy time. The combined idle is therefore about half of what either stream would produce alone, and the fill/drain cost $(P-1)(F+B)$ is shared across $2M$ micro-batches of work (the two streams double the throughput of the pipe for the same warm-up cost).

---

## 7. The bubble math, collected

| Schedule | Bubble ratio ($P$ stages, $M$ micro-batches) | Memory per stage |
|---|---|---|
| GPipe (fill–drain) | $(P-1)/(M+P-1) \approx (P-1)/M$ | $O(M)$ activations |
| 1F1B | $(P-1)/(M+P-1) \approx (P-1)/M$ (blocking) | $O(P)$ activations |
| Interleaved ($v$ chunks) | $\approx (P-1)/(vM)$ | $O(P)$ + $v\times$ transfers |
| **DualPipe** | **$(P-1)/(2M)$ — halved** | $O(P)$, two streams share recompute |

Worked example at DeepSeek-V3-ish scale ($P = 8$ stages, $M = 40$ micro-batches):

| Schedule | Bubble |
|---|---|
| GPipe / 1F1B | $7/40 = 17.5\%$ |
| Interleaved $v=2$ | $7/80 = 8.75\%$ |
| **DualPipe** | **$7/80 = 8.75\%$** + All-to-All hidden behind compute |

Two caveats on reading this table:

1. **The formula understates DualPipe.** The $(P-1)/(2M)$ term counts only the remaining *structural* idle. The bigger practical win is that the All-to-All stalls (Section 5) are hidden behind opposing-stream compute — a benefit no bubble formula captures, because it depends on the ratio of network latency to GEMM time.
2. **The formula assumes perfect balance.** Every stage must take the same $F$ and $B$. A stage with a heavier layer mix (more MoE layers, more experts) becomes the critical path and re-introduces idle everywhere else. This is why real deployments tune stage boundaries against measured per-layer times, not layer counts.

---

## 8. Why DualPipe is hard — the engineering surface

DualPipe is not a flag you flip; it requires:

1. **Model chunking** — each stage holds two halves of its layers (one per stream), so memory layout and the forward/backward graph must support dual streams without aliasing.
2. **A hand-built schedule** — the exact ordering of which stream computes which layer when, per stage, is a constraint-satisfaction problem (dependencies + communication + compute durations). DeepSeek-V3's schedule is hardcoded, not auto-derived: for each (stage, stream, layer) it fixes a wall-clock slot such that both streams' forward/backward chains and all-to-all deadlines hold. Getting this wrong deadlocks the two streams (each waiting on the other's buffer).
3. **Forward/backward recomputation** — to fit both streams' activations, DualPipe recomputes the forward in the backward (gradient checkpointing per stream), so the memory for two streams stays near one stream's worth. This is the same trick `training/pretrain.py` uses on one device (`use_checkpoint=config.grad_checkpoint`), promoted to a scheduling invariant.
4. **Collective tuning** — All-to-All must be sized to fit inside the compute window of the opposing stream's layer; if comms exceed compute, the bubble returns. This couples kernel choice, MoE expert count, and network topology.

This is the engineering surface that makes DualPipe a *distributed-systems* project, not a model-architecture one. It is orthogonal to MLA/MoE/MTP — you could train the same model with plain 1F1B and lose the 2× bubble, or with DualPipe and gain it, on the *same* architecture.

---

## 9. Why this repo skips it — the single-GPU loop

DualPipe's value is realized only when (a) the model is split across ≥2 GPUs and (b) MoE All-to-All is a measurable stall. This repo's `training/pretrain.py` is an explicit **single-GPU** loop: one device, `torch.compile(max-autotune)`, gradient checkpointing, no `torch.distributed` process group, no `all_to_all`. The MoE experts all live on the one device, so routing is a `stacked`/`triton_grouped` local dispatch (see [12 Triton Kernels](../concepts/kernels-and-ops.md)), not a cross-node collective. There is nothing to pipeline.

The choice is deliberate and documented in the design goals (see [02 Model Architecture](../concepts/foundations.md) §Design Goals): *single-GPU training* and *one file explains the full train path* are both portfolio-defensible properties, and both are incompatible with pipeline parallelism. DualPipe is documented here so the portfolio covers the full DeepSeek-V3 distributed story; it is the natural next chapter if the model were scaled to a size that no longer fit on one device.

### 9.1 What the single-GPU loop actually does

The entry point is deliberately bare (`training/pretrain.py:main`):

```python
    trainer = Pretrainer(config)
    if args.resume is not None:
        trainer.load_checkpoint(int(args.resume))
    trainer.train()
```

No process group, no rank, no world size — just construct, optionally resume, run. `Pretrainer.__init__` (`training/pretrain.py:Pretrainer.__init__`) pins the device the same way:

```python
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

Every tensor moves to that one device (`tokens.to(self.device, non_blocking=True)`), and the training loop (`training/pretrain.py:Pretrainer.train`) is a plain sequential loop over a `DataLoader`:

```python
        while global_step < self.config.max_steps:
            for tokens, targets in tqdm(loader):
                if global_step >= self.config.max_steps:
                    break
                tokens = tokens.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                metrics = self.train_step(tokens, targets, global_step)
```

There is no stage, no micro-batch plumbing, no send/recv: the *whole 18-layer model* (2 dense SwiGLU + 16 MoE layers) is one module on one device, and the only parallelism inside a step is whatever the kernel scheduler and `torch.compile` extract from independent tokens.

### 9.2 Gradient accumulation: the single-GPU stand-in for micro-batching

Pipeline parallelism chops the global batch into $M$ micro-batches for *spatial* reasons (they must flow between stages). This repo chops the batch for *temporal* reasons, via gradient accumulation. The config (`training/pretrain.py:TrainingConfig`) defaults `batch_size = 8`, `gradient_accumulation_steps = 4`; the canonical YAML sets `micro_batch_size: 8` and `gradient_accumulation_steps: 4` with `max_seq_len: 2048`, giving an effective optimizer step of $8 \times 4 \times 2048 = 65\,536$ tokens. `train_step` (`training/pretrain.py:Pretrainer.train_step`) tracks the accumulation boundary:

```python
    def train_step(self, tokens: torch.Tensor, targets: torch.Tensor, micro_step: int) -> Optional[Dict[str, Optional[float]]]:
        is_opt_step = (micro_step + 1) % self.config.gradient_accumulation_steps == 0
```

… and divides the loss so the accumulated gradients average correctly:

```python
                loss = main_loss / self.config.gradient_accumulation_steps
```

The optimizer (`optimizer.step()`, `scheduler.step()`, `zero_grad()`) runs only when `is_opt_step` is true — exactly the "one optimizer step per $M$ micro-batches" rule a pipeline stage follows, minus the inter-stage communication. Gradient accumulation is what a pipeline would do *within* one stage; the pipeline adds the cross-stage dataflow on top.

### 9.3 The single-GPU optimization levers, mapped

| Pipeline parallelism provides… | This repo provides… |
|---|---|
| More devices (memory + FLOPs) | one device sized to fit: 411.6M params, BF16, ~185M active/token |
| Micro-batch flow between stages | gradient accumulation inside `train_step` |
| Recompute to bound activation memory | `use_checkpoint=config.grad_checkpoint` |
| Cross-stage latency hiding | `torch.compile(mode="max-autotune")` + kernel fusion |
| Cross-node expert routing | local `stacked` dispatch (`models/moe.py:DeepSeekMoE._routed_forward_stacked`) or opt-in `triton_grouped` (`models/moe_triton.py:triton_grouped_moe_dispatch`) |

Every row of that table is a *within-device* substitute for a *between-device* mechanism. If the model outgrew one A100, the natural port is the V3 stack this chapter describes — but that is explicitly out of scope for this reproduction (see the banner and [Design Goals](../concepts/foundations.md)).

---

## 10. Pitfalls

- **Bubble math assumes balanced stages.** The formulas divide by $P$ stages of equal $F+B$. This repo's layer mix is 2 dense + 16 MoE layers — an 18-layer model split into, say, 8 stages cannot be split evenly in *compute* (the dense SwiGLU layers cost ~1.3M FLOPs/token each vs ~8.8M for the MoE layers at canonical dims, per [04 DeepSeekMoE](../concepts/moe-mtp.md)). Real stage boundaries must be tuned against measured per-layer times, or the slowest stage sets the global clock and the bubble formula underestimates the true idle.
- **Exact vs asymptotic formulas.** $(P-1)/M$ is the large-$M$ limit of $(P-1)/(M+P-1)$. At $M = P$ they disagree by nearly 2× (see the $P=4, M=4$ check in Section 2). Quote the exact form for small $M$; the asymptotic form hides a real bubble.
- **The $F/B$ ratio cancels in the fraction, but not in wall time.** The bubble *fraction* is ratio-independent, but the absolute idle time grows with $B \approx 2F$. A schedule that looks fine in unit-time diagrams can still waste most of a training run if the backward drain is long — which is exactly what 1F1B's interleaving mitigates.
- **Overlap is only as good as the window it hides in.** DualPipe's $(P-1)/(2M)$ assumes the All-to-All fits entirely inside the opposing stream's compute window. If the network is slower than the GEMM (small expert GEMMs, congested topology), the stall reappears. The overlap claim is a *design target*, not a guarantee — and no GPU run of this repo (or of V3 at this scale) has measured it.
- **Batch-size semantics change under pipelining.** With pipeline parallelism, the effective batch is $M \times$ (per-stage micro-batch size) $\times$ sequence length, and the optimizer step happens once per $M$ micro-batches. Hyperparameters that are step-indexed — the warmup/cosine schedule, the MoE bias update cadence (`bias_update_every`), gradient clipping — all see a different step rhythm than the single-GPU loop's. The μP LR scaling in `training/pretrain.py:Pretrainer.__init__` depends only on parameter count ($6.0e{-4}\sqrt{757226496/N} \to 8.14e{-4}$ base / $8.07e{-4}$ with MTP), so it is pipeline-agnostic — but every *count-based* schedule is not.
- **MoE bias updates become stale and per-stage.** In this repo, `_update_moe_bias` (`training/pretrain.py:Pretrainer._update_moe_bias`) feeds routing counts from the last forward back into each gate's bias every `bias_update_every` optimizer steps (see [04 DeepSeekMoE](../concepts/moe-mtp.md) §Bias-update). Under pipeline parallelism each stage sees a *different* micro-batch at a given wall-clock time; the "global" load-balance signal becomes a per-stage, time-shifted signal, which can drive the 20 experts' biases out of sync. The aux-loss-free feedback loop is only well-defined in the single-GPU setting.
- **Bidirectional schedules can deadlock.** The two counter-streams share stage buffers; a schedule that lets stream A fill stage $i$'s buffer while stream B waits on stage $i+1$'s buffer (and vice versa) deadlocks. This is why DeepSeek-V3 hardcodes the schedule instead of deriving it at runtime — and why a correct implementation needs buffer-accounting tests that a single-GPU repo simply has no reason to write.
- **Do not retrofit.** Adding `torch.distributed` to this repo would break design goal #4 (CPU-testable: all correctness tests run without CUDA or a process group, see [11 Operations and Testing](../concepts/kernels-and-ops.md)). The 199-test suite assumes a single process and deterministic single-device numerics. Pipeline parallelism here is a *documented* technique, not a TODO.

---

## 11. Check your understanding

**Q1.** With $P = 8$ stages and $M = 40$ micro-batches, what are the bubble fractions for (a) GPipe/1F1B, (b) interleaved 1F1B with $v = 2$, (c) DualPipe? What does DualPipe provide beyond the interleaved number?

<details><summary>Answer</summary>

(a) $(P-1)/M = 7/40 = 17.5\%$; (b) $(P-1)/(vM) = 7/80 = 8.75\%$; (c) $(P-1)/(2M) = 7/80 = 8.75\%$. DualPipe's *additional* win is orthogonal to the bubble formula: it hides the All-to-All dispatch of each stream behind the opposing stream's compute, so the latency-bound collective stalls never idle the Tensor Cores. Interleaving pays for its bubble reduction with $v\times$ more activation transfers; DualPipe pays with schedule complexity.

</details>

**Q2.** Why does 1F1B beat GPipe if the bubble formula is identical?

<details><summary>Answer</summary>

Memory. GPipe holds $O(M)$ activation sets per stage; 1F1B holds $O(P)$. At a fixed memory budget, 1F1B can run a much larger $M$, and a larger $M$ is what actually drives the $(P-1)/(M+P-1)$ bubble down. 1F1B converts a memory bound into a bubble reduction.

</details>

**Q3.** The repo has no pipeline parallelism — what plays the role of micro-batching in `training/pretrain.py:Pretrainer.train_step`, and where does the optimizer step fire?

<details><summary>Answer</summary>

Gradient accumulation: `is_opt_step = (micro_step + 1) % gradient_accumulation_steps == 0`, with the loss divided by `gradient_accumulation_steps` so averaged gradients accumulate. `optimizer.step()`, `scheduler.step()`, and `zero_grad()` fire only when `is_opt_step` is true — the same "one optimizer step per $M$ micro-batches" cadence a pipeline stage uses, without any inter-stage dataflow.

</details>

**Q4.** The banner says "paper-spec only — not implemented." What single code fact makes that checkable?

<details><summary>Answer</summary>

`training/pretrain.py:main` constructs one `Pretrainer` and calls `trainer.train()` in one process, with `self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")` — no `init_process_group`, no rank/world-size, no `all_to_all`, and a `grep` for those symbols across the repo returns nothing. Every schedule in this chapter is documentation of the V3 algorithm, not a code path in this repository.

</details>

---

## References

- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) — §1.3.2 DualPipe, §2.3.4 all-to-all dispatch
- [Training](../training.md) — the single-GPU loop that stands in for pipelining here
- [MoE & MTP](../concepts/moe-mtp.md) — expert routing that DualPipe would dispatch across nodes
