# Compare the complete vocabulary

Run from the repository root. Use the downloaded Q4 model, one RTX 3090 and at least 6 GB of temporary space for score exports. Run the two scorers sequentially with the serving container stopped. The reference uses the publisher's unmodified x86-64 CUDA 13 portable llama.cpp libraries. The evaluation-only NInfer patch exports the scores used by its perplexity tool.

## Prepare the input

Install Python 3.11+ with `numpy` and Hugging Face `tokenizers`. Create a fresh output directory per run because the native evaluator appends to its output files.

```sh
out=$(mktemp -d "$PWD/results-inkling.XXXXXXXX")
curl -fsSL https://huggingface.co/datasets/ggml-org/ci/resolve/927b3642933080f1b0e811e2f916e14c292992f9/wikitext-2-raw-v1.zip -o "$out/wiki.zip"
python3 bench/quality/make-excerpt.py "$out/wiki.zip" models/frontend/tokenizer.json "$out/excerpt.txt"
```

The script checks the archive and the exact excerpt hash. The input is the first 4,210 tokens from the WikiText-2 raw test split. [Dataset credits](../../licenses/README.md)

## Export the optimized scores

```sh
docker build --platform linux/amd64 --build-arg SCORE_CAPTURE=1 -t inkling-q4-score .
docker run --rm --gpus '"device=0"' --user "$(id -u):$(id -g)" \
  -v "$PWD/models:/models:ro" -v "$out:/out" \
  -e INKLING_MMA_PREFILL=1 -e INKLING_Q8_GROUP=1 -e INKLING_PACK_EXPERTS=1 \
  -e INKLING_SCORE_DUMP=/out/optimized --entrypoint ninfer-perplexity inkling-q4-score \
  /models/native/Inkling-q4.ninfer --text /out/excerpt.txt --context 512 \
  --disjoint --kv-dtype fp16 --output /out/report
```

The tool writes eight independent 512-token windows in `optimized.ids` and all next-token logits in `optimized.bf16`. Each window supplies 511 predictions.

## Score the same token IDs in vanilla Q4

On an x86-64 Linux host with a C++17 compiler, a compatible CUDA 13 runtime and NVIDIA driver, download the pinned release and its matching headers. [Unsloth reference release](https://github.com/unslothai/llama.cpp/releases/tag/b12588-mix-c0c3de8)

```sh
release=https://github.com/unslothai/llama.cpp/releases/download/b12588-mix-c0c3de8
curl -fsSL "$release/app-b12588-mix-c0c3de8-linux-x64-cuda13-portable.tar.gz" -o "$out/ref.tgz"
curl -fsSL "$release/llama.cpp-source-commit-883286076efe5b85fddb31d9cc7035a6b698ba9b.tar.gz" -o "$out/src.tgz"
(cd "$out" && echo '6c1985d2d75c9cc6b299891d557aca70843ba5ccb68f14863a50687048cfdd48  ref.tgz' | sha256sum -c - &&
 echo '92ee8439f9ae51a40f15c168ac64e1d657bf8b08d301c96f8fb2d58625573607  src.tgz' | sha256sum -c -)
mkdir "$out/ref" "$out/src"
tar xf "$out/ref.tgz" -C "$out/ref"
tar xf "$out/src.tgz" -C "$out/src" --strip-components=1
lib=$(dirname "$(find "$out/ref" -name libllama.so.1 -print -quit)")
c++ -std=c++17 -I"$out/src/include" -I"$out/src/ggml/include" bench/quality/vanilla-logits.cpp \
  -L"$lib" -Wl,-rpath-link,"$lib" -l:libllama.so.1 -l:libggml.so.1 -o "$out/vanilla-logits"
LD_LIBRARY_PATH="$lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" "$out/vanilla-logits" \
  models/Inkling-2-Mini-UD-IQ4_XS.gguf "$out/optimized.ids" "$out/vanilla.f32" "$lib"
python3 bench/quality/compare-logits.py "$out/vanilla.f32" "$out/optimized.bf16"
```

The comparison checks literal target token IDs and vocabulary size before reporting top-token matches, the reference rank of disagreements, perplexity, KL divergence and total variation. The original run covered 4,088 predictions over 200,058 vocabulary entries. [Measured results](../../README.md#measured-results)
