# Theoretical Ceilings: Inkling 2 Mini (UD-IQ4_XS) on RTX 3090 (250W cap)

## 1. Measured Hardware (not spec sheet)

Measured on idle RTX 3090 (24 GiB GDDR6X, compute cap 8.6, driver `580.196.07`, CUDA 13.5 `nvcc`, `gpu-unlock` active, **`nvidia-power-limit.service` enforcing 250 W cap**):

| Metric | Measured | Spec / Notes |
|---|---:|---|
| D2D copy bandwidth (512 MiB, CUDA events) | **706.2 GB/s** median (645–782 GB/s) | 936.2 GB/s spec; 9501 MHz mem, 1770 MHz SM, 113 W |
| Sustained GEMM 4096³ (FP16 / BF16, FP32 acc) | **67.2 / 79.1 TFLOPS** | 130–155 W, 840–1080 MHz SM |
| Sustained GEMM 8192³ (FP16 / BF16, FP32 acc) | **59.3 / 59.8 TFLOPS** | 240–246 W (hits 250 W cap), 1020–1035 MHz SM |
| Practical compute (MFU = 0.50 × 59.8 TFLOPS) | **29.9 TFLOPS** | Power-cap limited at 250 W (390 W unlocked was ~137 TFLOPS) |
| Usable VRAM cap (with desktop ~1.5 GiB) | **≤ 23.0 GiB** | 24,576 MiB total |

## 2. Exact Model Component Sizes (`Inkling-q4.ninfer` + `proposal-q4.q4`)

Inkling 2 Mini is a looped MoE transformer: **24 stored layers** executed as **48 visits** (6 prelude `0..5` × 1 + 12 core `6..17` × 3 + 6 coda `18..23` × 1), 128 routed experts top-8, hidden 2304, 24 Q / 8 KV heads of dim 128, 8 full-attention visits (layers 5, 11, 17, 23 across passes), 40 sliding-window visits (window 1024), vocab 200,058.

| Component | Stored GB | Effective / Target Step (GB) | Format |
|---|---:|---:|---|
| Prelude + Coda dense (12 layers × 1) | 0.3415 | 0.3415 | Q8_0 / Q6_K / Q5_K / FP32 |
| Core dense non-loop (12 layers × 3 visits) | 0.3620 | 1.0860 | Q8_0 / Q6_K / Q5_K / FP32 |
| Core loop controllers (12 layers × 2 passes) | 0.1137 | 0.2274 | FP32 |
| **Total dense weights / target forward** | **0.8172** | **1.6549** | |
| Routed experts pool (48 visits) | 14.0363 | 27.7966 pool → **1.7373** (M=1, 8/128) | NVFP4 (20L), Q5/Q4/IQ4 (4L) |
| Output LM head (200,058 × 2304) | 0.3799 | **0.3799** | Q6_K |
| **Target M=1 total (ctx=0)** | | **3.7721 GB** | |
| DSpark 3-layer backbone (once/round) | 0.1474 | 0.1474 | Q4_0 |
| DSpark `proposal-q4.q4` head (once/round) | 0.2605 | 0.2605 (0.0427 @ 32K vocab) | Q4_0 |
| DSpark Markov table `a.m` + `a.weight` (×k/round) | 0.1181 | **0.1181 × k** (0.0325 × k @ 32K) | BF16 + FP32 |
| Vision tower + Embedding | 1.1314 | — | BF16 + Q8_0 |

### KV Cache Sizing (8 full-attention visits + 1 DSpark full-attn layer)

- **F16 KV (current `ninfer-3090`)**: `8 × 2 × 8 × 128 × 2 = 32,768 B/tok` (32 KiB/tok) → **2.68 GB** at 81,920 tokens (+ ~224 MiB sliding rings + ~335 MiB DSpark KV). Fits `c=1` at 80K (~21.7 GiB VRAM).
- **MXFP8 KV (`ninfer-all` default, trained format)**: `16,896 B/tok` (16.5 KiB/tok) → **1.38 GB** at 81,920 tokens (**−1.30 GB VRAM**, halves full-attention KV read bandwidth!).

## 3. Decode Ceilings (c1, at measured 706.2 GB/s steady bandwidth)

| Architecture / Lever State | k=0 (a=1.0) | k=1 (a=1.65) | k=2 (a=2.25) | k=3 (a=2.75) | k=4 (a=3.10) |
|---|---:|---:|---:|---:|---:|
| **Current (`ninfer-3090`: per-row F16 KV, 200K draft)** — ctx=0 | 187.2 | 196.6 | 209.8 | **212.9** | 206.9 |
| Current — ctx=4K | 180.8 | 188.1 | 199.2 | **201.1** | 194.5 |
| Current — ctx=32K | 145.7 | 144.3 | **147.2** | 144.7 | 137.3 |
| Current — ctx=80K | **109.4** | 103.2 | 101.7 | 97.8 | 91.2 |
| **+ Shared-tile verify attention (1× KV read)** — ctx=32K | 145.7 | 166.4 | 183.8 | **190.5** | 187.8 |
| + Shared-tile verify attention — ctx=80K | 109.4 | 135.3 | 154.9 | 164.5 | **165.0** |
| **+ 32K pruned draft vocab (lossless)** — ctx=0 | 187.2 | 207.2 | 221.2 | **224.6** | 218.5 |
| + Shared-tile verify + 32K pruned draft — ctx=32K | 145.7 | 174.0 | 192.4 | **199.8** | 197.3 |
| + Shared-tile verify + MXFP8 KV + 32K pruned draft — ctx=32K | 163.2 | 188.6 | 205.4 | **211.1** | 207.0 |
| + Shared-tile verify + MXFP8 KV + 32K pruned draft — ctx=80K | 137.0 | 166.3 | 185.5 | **193.6** | 191.9 |

**Key theoretical insights:**
1. **Draft vocab overhead is massive**: at `k=2..4`, scanning all 200,058 rows of `proposal_head` (260.5 MB) + `k × a.m` (`k × 102.4 MB`) is **0.47–0.67 GB per round** (73% of total draft bytes!). Pruning the draft vocabulary to 32K/16K (while target still verifies all 200K rows losslessly) cuts draft bytes by **~2.5–3×**!
2. **Per-row verify attention kills long-context decode**: `attention_kernel<1>` launches `t = k+1` separate blocks that each re-read the entire KV cache (`(k+1) × 32 KiB × ctx`). At 80K ctx, per-row verify drops the ceiling from 165 tok/s to 91–102 tok/s (worse than turning speculation OFF!). Sharing KV loads across all `t ≤ 8` verify queries raises the 32K/80K ceiling by **+32% / +68%**!
3. **Verify expert union**: because 8/128 experts are routed per token across 48 visits, `k+1` tokens activate `1 - (120/128)^(k+1)` of the 27.8 GB pool (`17.6%` at `k=2`, `22.75%` at `k=3`, `27.6%` at `k=4`), so optimal `k` peaks around `k=2..3` (and lower if acceptance drops).

## 4. Prefill, TTFT, and Startup Ceilings

- **Prefill (compute bound @ 29.9 TFLOPS practical BF16 at 250 W)**:
  - `active_params = 5.4B` (3 passes) → `10.8 GFLOP/tok` linear + `4 × 8 × 24 × 128 × (L/2)` attention:
  - **4K**: **2,718 tok/s**
  - **32K**: **2,409 tok/s**
  - **80K**: **2,017 tok/s**
- **Short-prompt TTFT floor**: 1 forward pass (`3.77 GB / 706 GB/s = 5.3 ms`) + tokenize/schedule/HTTP (~15–30 ms) ≈ **20–35 ms**.
- **Startup floor**: `16.66 GB / ~3.0 GB/s` (warm page cache) + CUDA graph capture (~0.5 s) ≈ **6–8 s**.
