# DeepSeek-v3-Lite — Documentation Index

> **Start here** after [[Docs/00_Getting_Started]]. Textbook-style reference for a pure-PyTorch DeepSeek-V3 LLM. Project postcard: [[Reference]].

---

## Learning Path

| Step | Chapter | Learner Outcome |
|---|---|---|
| 00 | [[Docs/00_Getting_Started\|00 Getting Started]] | Project overview, installation, smoke tests |
| 01 | [[Docs/01_Foundations\|01 Foundations]] | DeepSeek lineage (V1 $\rightarrow$ V2 $\rightarrow$ V3) |
| 02 | [[Docs/02_Model_Architecture\|02 Model Architecture]] | Full model topology, parameter budget, specs |
| 03 | [[Docs/03_Multi_Head_Latent_Attention\|03 MLA]] | Low-rank KV compression, decoupled RoPE |
| 04 | [[Docs/04_DeepSeekMoE\|04 DeepSeekMoE]] | Fine-grained expert routing, shared experts |
| 05 | [[Docs/05_Multi_Token_Prediction\|05 MTP]] | Multi-token prediction depth & speculative decoding |
| 06 | [[Docs/06_FP8_Mixed_Precision\|06 FP8 Mixed Precision]] | E4M3/E5M2 quantization & block scaling |
| 07 | [[Docs/07_DualPipe_Parallelism\|07 DualPipe]] | Bidirectional pipeline parallelism & overlap |
| 08 | [[Docs/08_Training_Pipeline\|08 Training]] | Loop, AdamW, cosine schedule, loss balancing |
| 09 | [[Docs/09_Data_Pipeline\|09 Data Pipeline]] | Dataset mix, tokenization, binary mmap |
| 10 | [[Docs/10_Inference_and_Serving\|10 Inference]] | Autoregressive sampling, MLA KV decompression |
| 11 | [[Docs/11_Operations_and_Testing\|11 Operations]] | Pytest suite, checkpoints, VRAM budget |
| 12 | [[Docs/12_Triton_Kernels\|12 Triton Kernels]] | Fused FP8 GEMM & MoE dispatch Triton kernels |
| 13 | [[Docs/13_Portfolio_Comparison\|13 Portfolio Comparison]] | Architecture comparison vs LLaMA-3, Mamba-3, HyMo |

---

## Learner Routing

| Question | Read First |
|---|---|
| I'm new — what is this? | [[Docs/00_Getting_Started]] |
| Model evolution (V1 $\rightarrow$ V2 $\rightarrow$ V3) | [[Docs/01_Foundations]] |
| How does MLA compress KV cache? | [[Docs/03_Multi_Head_Latent_Attention]] |
| How does DeepSeekMoE route tokens? | [[Docs/04_DeepSeekMoE]] |
| How does Multi-Token Prediction (MTP) work? | [[Docs/05_Multi_Token_Prediction]] |
| FP8 quantization & block scaling | [[Docs/06_FP8_Mixed_Precision]] |
| DualPipe pipeline parallelism | [[Docs/07_DualPipe_Parallelism]] |
| Train loop & loss balancing | [[Docs/08_Training_Pipeline]] |
| Dataset mixture & tokenization | [[Docs/09_Data_Pipeline]] |
| Autoregressive decode & serving | [[Docs/10_Inference_and_Serving]] |
| Tests & checkpointing | [[Docs/11_Operations_and_Testing]] |
| Triton kernels | [[Docs/12_Triton_Kernels]] |
| Portfolio comparison | [[Docs/13_Portfolio_Comparison]] |

---

## External Index

| Resource | Location |
|---|---|
| Project Reference | [[Reference]] |
