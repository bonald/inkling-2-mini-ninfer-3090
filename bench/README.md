# Reproduce the measurements

## Decode throughput

The released figures used the first eight prompts in the pinned HyperQwen setup-B cohort. Its file digest matches the measured input. Run the following from the repository root against a warm, otherwise idle RTX 3090. Each run excludes a 32-token warm-up, uses temperature 1, seed 42 and unrestricted sampling, and requests eight consecutive 1,024-token streams with EOS ignored. The harness verifies the final SSE event, text and completion-token counts.

```sh
mkdir -p results
curl -fsSL https://raw.githubusercontent.com/syv-ai/HyperQwen/ce9615c4f2309be727407918a6c979bd67968069/bench/prompts_real.jsonl -o results/prompts.jsonl
echo '27da4fd8b2dcc133b98cec54bc337c34f5acb1b6372cc2fea1d8a92da9c78650  results/prompts.jsonl' | sha256sum -c -
python3 bench/decode.py results/prompts.jsonl > results/decode.jsonl
```

Decode tokens/s is the reciprocal of the arithmetic mean of the eight request-level times per output token after the first text. End-to-end speed divides total completion tokens by total request wall time. The control was Qwen3.8-27B on HyperQwen, timed on the same card, through the same gateway, using the same prompts and settings. Each model uses its own tokenizer.

## Answer and token comparisons

Use the published CUDA 13 llama.cpp build and original Q4 GGUF as the reference. The full-vocabulary instructions below give the archive and its digest. With the optimized service stopped, launch the publisher's `llama-server` from its release directory. Use `--alias vanilla-q4 -m /path/to/models/Inkling-2-Mini-UD-IQ4_XS.gguf -ngl 99 -c 65536 -np 1 --port 8081` and leave speculation disabled. Run the reference capture. Then stop the reference, start the optimized server on port 8080 on the same GPU, and run the candidate capture and comparison. The candidate reuses the literal token prefixes from the reference JSON.

```sh
mkdir -p results
python3 bench/compare-quality.py capture http://127.0.0.1:8081 vanilla-q4 results/vanilla.json
python3 bench/compare-quality.py capture http://127.0.0.1:8080 inkling-2-mini-q4 results/optimized.json --reference-inputs results/vanilla.json
python3 bench/compare-quality.py compare results/vanilla.json results/optimized.json
```

This harness covers 95 fixed contexts, six free continuations of up to 80 tokens, and 12 checkable questions under greedy and two seeded sampling conditions. The separate full-vocabulary comparison uses WikiText-2 and evaluates 4,088 positions. [Full-vocabulary scoring](quality/README.md)

To repeat the near-limit retrieval timing on the optimized server, run `python3 scripts/check-context.py`. It checks the configured context size, submitted and generated token counts, and the answer to a question about the beginning of the prompt.
