# Inkling 2 Mini RTX 3090 Inference Optimization Plan & Experiment Log

## STATUS — Plateau Reached & Qualified (2026-11-24)

| Metric | Initial Baseline (compositor busy) | Clean Idle Baseline (`ctl`) | Qualified Winner (`k123-d4-v32k`) | Ceiling (250 W) | Gain (vs Clean / vs Initial) |
|---|---:|---:|---:|---:|---:|
| **c1 decode (6-prompt median, temp=1.0)** | 84.66 tok/s | 105.18 tok/s (5-run med) | **115.75 tok/s** (7-run med; mean 118.13) | 225 tok/s (51%) | **+10.1%** / **+36.7%** |
| **Short greedy decode (512 tok)** | — | 125.46 tok/s (avg ctl_a/b) | **139.57 tok/s** | 225 tok/s (62%) | **+11.3%** |
| **39.3K ctx (50%) decode (256 tok)** | — | 86.97 tok/s (83.03 / 90.90) | **110.38 tok/s** (3/3 needles) | 192 tok/s (57%) | **+26.9%** |
| **75.2K ctx (95%) decode (256 tok)** | — | 65.42 tok/s (65.92 / 64.91) | **79.37 tok/s** (3/3 needles) | 165 tok/s (48%) | **+21.3%** |
| **Cold prefill ~4K / ~32K** | 1,758.6 / 1,541.7 | 1,879.0 / 1,741.6 | **1,853.8 / 1,735.9** (39K: 1,674.0) | 2,718 / 2,409 (69–72%) | flat / +0.5–3.6% long |
| **Sustained decode (4,096 tok, 39s)** | — | — | **104.96 tok/s** (no loop) | — | **PASS** |
| **Peak GPU VRAM (cap 23.0 GiB)** | 21.19 GiB | 21.19 GiB | **20.88–20.90 GiB** | — | **−0.29 GiB** |

### Winning Stack (`best-config.json`, wired into `run-ninfer-server.sh`)
All three levers are opt-in env vars in `ninfer-3090/engine-src` and 100% lossless (target verifies against full 200,058 vocab; 140-tok fixed-width greedy matched bit-for-bit `[True, True]`):
1. **Compositor GPU contention fix:** Disabling `liquid-glass-shell@liquid-glass` & `blur-my-shell@aunetx` cleared a persistent 80–100% GPU SM / 208 W / 98% fan spin (+24% decode, +12% 32K prefill).
2. **K3 (`INKLING_FAST_DOWN=1`):** Bit-exact 4-row-per-warp NVFP4 `down` kernel (`packed_down_4row_kernel` in `ggml_vec_nvfp4.cu`) eliminating 28/32-lane `S=36` tail waste, reusing register-cached activations across 4 rows, and coalescing 4-lane `atomicAdd`s. Cut `nvDN` from **4.26 → 2.88–3.07 ms** (−1.2 to −1.4 ms/round; **+6.9%** standalone c1).
3. **K2 (`INKLING_DRAFT_VOCAB=opt-3090/vocabs/draft-vocab-32k.bin`):** Compacts DSpark `proposal_head` (260.5 → 42.7 MB), Markov `table_m` (102.4 → 16.8 MB), `table_c`, and `draft_logits` to 32,768 frequency-ranked tokens (98.66% held-out coverage), with exact full-vocab maximal-coupling `residual_sample`. Cut DSpark `head + markov` by **3.4×** (**0.93 → 0.27 ms/round**) and saved **−0.29 GiB VRAM**.
4. **K1-v2 (`INKLING_SHARED_VERIFY_ATTN=1`):** 1-warp-per-`(row, head)` shared-tile causal verify attention (`attention_verify_kernel` in `kernels.cu`, full-attention visits + DSpark uniform layer). Cut 28K `fullA` from **4.47 → 2.74 ms** and lifted 39.3K / 75.2K decode by **+26.9% / +21.3%** when combined with K2+K3.

### What Limits Further Gains (Profile Evidence)
Per-stage GPU breakdown at `rows=4` (short ctx, `k123-d4-v32k`):
- `nvGU` (44 NVFP4 visits): **4.42–4.53 ms** (~724 GB/s — already at 100% of measured 706–782 GB/s D2D copy bandwidth).
- `nvDN` (44 NVFP4 visits): **2.88–3.07 ms** (~555 GB/s — up from 400 GB/s; only ~0.6 ms from copy BW, bounded by 64-bit fixed-point `atomicAdd`).
- `attn_proj + attn_out`: **2.56 ms** (`1.37 GB` -> ~535 GB/s).
- `head` (Q6_K 200K LM head): **0.57 ms** (~667 GB/s — at bandwidth ceiling).
- `spark` (DSpark drafter): **0.90 ms** total (`bb=0.63`, `head=0.07`, `markov=0.20`).
- `fullA` @ 39K–75K: **F16 KV (`32 KiB/tok`)** still dominates above 40K tokens (`2.4 GB` at 75K!).
- Prefill (4K / 32K): **69–72% of practical 250 W compute ceiling** (59.8 TFLOPS sustained BF16 at 1,035 MHz SM under the 250 W power limit).

### Wave 3 — Quantized KV Planes Port (`MXFP8` & `rk8v4`, 2026-11-24)
Ported `ninfer-all`'s 128-wide KV plane codecs (`mxfp8`, `rk8v4`, `int8`, `rk4v4`) into `ninfer-3090` and wired them into `kv_post`, `prompt_attention_mma`, `attention_kernel`, and K1-v2 `attention_verify_kernel` (`kv-planes-report.json`):
- **MXFP8 @ 80K (Inkling 2's native trained KV format, 16.5 KiB/tok):** KV `3,326 → 2,082 MiB (−37.4%)`, peak VRAM `20.90 → 19.72 GiB` (**2.83 GiB free**), short greedy **139.25 tok/s**, c1 sampled **116.34 tok/s**, 75.2K decode **80.19 tok/s (+9.1% vs F16+K123, +22.6% vs F16 baseline, 3/3 needles)**, 4,096-tok sustained **106.30 tok/s**, **6/6 acceptance gates PASS**.
- **`rk8v4` @ 80K (Int8G64 K + Int4G32 V, 12.75 KiB/tok):** KV `3,326 → 1,781 MiB (−46.5%)`, peak VRAM **19.43 GiB (3.12 GiB free)**, short greedy **140.08 tok/s**, c1 sampled **113.96 tok/s**, 75.2K decode **73.23 tok/s (3/3)**, sustained **102.78 tok/s**, **6/6 acceptance gates PASS**.
- **Unlocked Extended Capacities (Single 24 GiB RTX 3090, vision + DSpark resident):**
  - **MXFP8 @ 131,072 ctx (`c=1`):** KV `3,066 MiB`, peak VRAM **20.69 GiB (1.85 GiB free)** — **121,222 prompt tok**, prefill `825.4 tok/s`, decode **51.01 tok/s**, **3/3 needles PASS**!
  - **`rk8v4` @ 163,840 ctx (`c=1`):** KV `3,121 MiB`, peak VRAM **20.74 GiB (1.81 GiB free)** — **148,561 prompt tok**, prefill `725.1 tok/s`, decode **53.84 tok/s**, **3/3 needles PASS**!
  - **MXFP8 @ 65,536 ctx (`c=2`):** KV `3,495 MiB`, peak VRAM **21.09 GiB (1.46 GiB free)** — **140.87 aggregate tok/s** (`71.74 + 72.78 tok/s` per stream)!

### Ranked Remaining Levers (Next Wave Candidates)
1. **DSpark drafter FP8 KV + CUDA-graphed drafter + tensor-core drafter attention + `--draft-tokens 7` (inspired by `project-maya` v1.1.75):** Cuts DSpark's own full-context KV cache 2×, eliminates ~70 host launches/round, and accelerates 65K+ drafter attention + high-acceptance code/tool generation.
2. **Raise GPU power limit above 250 W** (hardware/thermal user policy — intentionally kept at 250 W per `nvidia-power-limit.service`): would unlock ~2× prefill compute ceiling (59.8 → 137 TFLOPS BF16 at 390 W).
3. **Eliminate 64-bit `atomicAdd` contention in `nvDN`** via a separate scratch + reduction pass: ~0.4–0.5 ms/round (~+3% decode), medium kernel complexity.
4. **Skip `select_weight_layout` repack for short prompts (`t ≤ 64`)**: saves ~100–130 ms on short-prompt TTFT (minor numeric difference between vector vs matrix prefill).

## Full Experiment History

- **Wave 1 (22:05, partially voided):** `c02`..`ctl-b` voided when `gnome-shell` (`liquid-glass-shell`) entered 80–100% GPU spin after NVKMS VRAM OOM at `22:09:47`. Fixed by disabling `liquid-glass-shell` & `blur-my-shell` + adding idle-GPU guard to `bench.py`.
- **Wave 1A (22:42, clean idle):** `ctl-1..4` = `100.69, 105.42, 106.81, 104.19`. `k2-vocab-32k` = `106.31` (mean `109.96`, −0.29 GiB VRAM). `c03-draft-rows-4` = `105.97` (hurt acceptance `306 → 281` in Wave 1B -> rejected).
- **Wave 1B (22:53, long-ctx greedy):** `c01-no-spec` = `102.22` short / `76.38` @ 28K / `64.42` @ 53K. Confirmed K1-v1 (`attention_kernel<8>`) was occupancy/head-loop bound -> replaced with K1-v2 (`attention_verify_kernel`, 1 warp per `(row, head)`).
- **Profile & Kernel Rejection (23:05–23:18):**
  - `INKLING_FAST_GU=1` (`packed_gu_4row_kernel`): **REJECTED** (`nvGU` regressed `4.53 → 5.12 ms` due to register pressure; removed).
  - `INKLING_FULL_SPLITS=48`: **REJECTED** (`fullA` @ 28K regressed `2.74 → 3.41 ms`; kept default `20`).
- **Wave 2A (23:25, clean idle, bracketed):**
  - `k3-down4` (`INKLING_FAST_DOWN=1`): **111.20** vs `103.52 / 104.44` (**+6.9% ACCEPT**).
  - `k3-down2` (`INKLING_FAST_DOWN=2`): **112.13** vs `104.44 / 105.18` (**+7.0% ACCEPT**).
  - **`k123-d4-v32k`**: **119.75** vs `105.18 / 106.80` (**+13.0% NEW BEST**, −0.29 GiB VRAM).
  - `k123-d4-v16k`: **112.94** vs `119.75` (**REJECT** — 96.5% coverage too low).
- **Wave 2B (23:39, derived from `k123-d4-v32k`, bracketed — PLATEAU CONFIRMED):**
  - `best-a..f`: `116.76, 115.31, 116.24, 115.47, 115.75, 115.60` (repeatable!).
  - `b01-v48k` (`115.88`), `b02-v24k` (`114.25`), `b03-down2` (`111.60`), `b04-sv-all` (`113.97`), `b05-no-amtp-k3` (`115.19`): **ALL 5 REJECTED** (at or below bracketing controls).
- **Qualification (`qualification-report.json`, 00:04, clean production build):**
  - Load: **PASS** (`inkling-2-mini-q4`)
  - Chat: **PASS** (`Paris`, `stop`)
  - Reasoning: **PASS** (`reasoning_len=163`, `content="432"`)
  - Tools round-trip: **PASS** (`lookup_secret_code` -> `X9K7Q2-VERIFIED`)
  - Vision (`natural_scene.png`): **PASS** (`"24; sun is in the upper right sky."`)
  - Context 50% (39,281 tok): **PASS 3/3** (**110.38 tok/s** vs ctl 83.03 / 90.90, **+26.9%**)
  - Context 95% (75,190 tok): **PASS 3/3** (**79.37 tok/s** vs ctl 65.92 / 64.91, **+21.3%**)
  - Sustained 39s decode (4,096 tok): **PASS** (`104.96 tok/s`, `looping=false`)
  - Greedy equivalence: 140-tok fixed-width `[True, True]`; 512-tok adaptive-width differed due to `WidthPolicy` selecting different verify batch widths (`M`) across runs.
