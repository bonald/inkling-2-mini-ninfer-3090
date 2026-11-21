# Inkling 2 Mini on an RTX 3090

A standalone, single-GPU build of NInfer for Unsloth's **Inkling 2 Mini UD-IQ4_XS**. It serves text and images through an OpenAI-compatible API with a 65,536-token context. On an RTX 3090 it measured **125.94 decode tokens/s**, with **96.77% next-token agreement** against vanilla Q4.

## Run with Docker

Use a Linux x86-64 host with a 24 GB RTX 3090, a CUDA 13-compatible NVIDIA driver, the NVIDIA Container Toolkit, Docker Compose v2 or newer, and about 45 GB free for model preparation plus Docker build space. The server compiles from source for SM86. Preparation downloads and converts the weights on the CPU.

```sh
mkdir -p models
docker compose build prepare inkling
docker compose run --rm --user "$(id -u):$(id -g)" prepare
docker compose up -d inkling
curl -fsS http://127.0.0.1:8080/v1/models
```

```sh
curl -fsS http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"inkling-2-mini-q4","messages":[{"role":"user","content":"What is 17 times 6?"}],"temperature":0,"max_tokens":256}'
```

The port binds to localhost. The health check lists available models so it can run while the GPU is idle. NInfer unloads an idle model after 600 seconds and reloads it on demand. Set `INKLING_MODEL_DIR`, `INKLING_GPU`, `INKLING_PORT` or `BUILD_JOBS` to change the data directory, GPU index, port or compilation parallelism. To stop the service, run `docker compose down`. To check both text and images, run `python3 scripts/check-chat.py`.

For downloads alone, run `python3 scripts/download-model.py models` on Linux or macOS with Python 3.11+ and curl. Add `--verify-only` to check existing files offline. Every source file has a pinned URL and SHA-256, including the 15,733,755,232-byte Q4 GGUF. Preparation checks the derived drafter and verifies the native model's tensor bytes, configuration and tokenizer against the measured build before installation. Repeating the preparation command verifies an existing native model.

## What makes it fast

- **Fewer GPU launches.** The attention path groups up to five Q8 projections in one CUDA launch. A lossless layout change lets the GPU read the routed experts' NVFP4 weights more efficiently.
- **Cheaper speculative decoding.** DSpark proposes up to two tokens. Its 27 backbone projections and its output projection use smaller Q4_0 copies. A per-request controller uses measured time and acceptance to select zero, one or two proposals. The original Q4 target always verifies them with all eight selected experts and the full vocabulary.
- **Faster prompt processing.** A tensor-core attention kernel accelerates long inputs. In the native engine, a 65,504-token retrieval fell from 72.09 to 37.40 seconds. Greedy prompt caching uses reproducible boundaries.

## Measured results

Eight warm, sequential prompts generated 1,024 tokens each on the same RTX 3090. These are single-run gateway measurements of the source configuration packaged here. Independent repeat-run uncertainty was not measured.

| Engine | Decode tokens/s | End-to-end tokens/s |
| --- | ---: | ---: |
| Inkling Q4, preceding tuned llama.cpp build | 97.44 | 94.68 |
| **Inkling Q4, patched NInfer** | **125.94** | **123.87** |
| Qwen3.8-27B, HyperQwen comparison | 130.20 | 127.95 |

The quality reference used the publisher's original Q4 GGUF and unmodified CUDA llama.cpp with speculative decoding disabled. Eight disjoint 512-token WikiText-2 windows gave **3,956/4,088 identical top-token predictions (96.77%)**, scoring all 200,058 token logits at each position. Of the 132 disagreements, 127 selected vanilla Q4's runner-up and all 132 selected a top-five token. Perplexity was **9.25762** versus **9.23235** for vanilla Q4.

With free generation, three of six greedy 80-token continuations matched exactly. On 12 checkable questions, optimized Q4 scored **12/12 greedy and 23/24 sampled**, versus **11/12 and 23/24** for vanilla Q4 at the same 768-token budget. These tests measure token and answer fidelity on the stated samples. [Reproduce the benchmarks](bench/README.md)

## Credits and license

Thinking Machines Lab created Inkling and DSpark. Unsloth published the GGUFs. The runtime is iamwavecut's NInfer-all and its upstream contributors' work. This repository provides the RTX 3090 patch, preparation scripts and evaluations under Apache-2.0. Model and dependency licenses are listed separately. [Credits and third-party terms](licenses/README.md) · [License](LICENSE)
