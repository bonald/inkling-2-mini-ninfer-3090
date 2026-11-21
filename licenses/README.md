# Attribution and distribution

The local patches, scripts, documentation and Docker recipe are licensed under Apache License 2.0. Files derived from upstream retain their applicable notices. [License](../LICENSE) · [Project notices](../NOTICE)

| Component | Source and terms |
| --- | --- |
| NInfer-all | Apache-2.0. Maintained by iamwavecut, building on Neroued/ninfer, Don-Chad/ninfer-3090, ashalliants/ninfer-3090 and other contributors. [Upstream contributor credits](https://github.com/iamwavecut/ninfer-all/blob/395ce8b85627bcb2d53a1a81623088eab755443c/CONTRIBUTORS.md) |
| Inkling 2 Mini and DSpark | Thinking Machines Lab. [Model card and Apache-2.0 license](https://huggingface.co/thinkingmachines/Inkling-2-Mini/tree/ed40e7da625f594fc9c4990f5f66be24873c22d7) |
| Quantized model, projector and GGUF drafter | Unsloth. [Quantization release, Apache-2.0](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/tree/ea0c79b2ecd73669e774c8519b79bfabfddbc027) |
| GGUF reader, quantization routines and reference implementation | Georgi Gerganov, GGML and llama.cpp contributors, and Unsloth's mixed-source fork. [GGUF tooling license](gguf-py-MIT.txt) · [GGML license](ggml-MIT.txt) · [Reference release](https://github.com/unslothai/llama.cpp/releases/tag/b12588-mix-c0c3de8) |
| Throughput prompts | syv-ai/HyperQwen, Apache-2.0. [Benchmark cohort](https://github.com/syv-ai/HyperQwen/blob/ce9615c4f2309be727407918a6c979bd67968069/bench/prompts_real.jsonl) |
| WikiText-2 | Stephen Merity, Caiming Xiong, James Bradbury, Richard Socher and the Wikipedia contributors. The benchmark publishes aggregate statistics. Fetch the evaluation text separately under the terms in the [dataset card](https://huggingface.co/datasets/Salesforce/wikitext). |
| Additional reasoning questions | OpenAI GSM8K (MIT) and the Allen Institute for AI ARC-Challenge (CC BY-SA 4.0). The script downloads the original questions at evaluation time. [GSM8K](https://huggingface.co/datasets/openai/gsm8k/tree/740312add88f781978c0658806c59bc2815b9866) · [ARC](https://huggingface.co/datasets/allenai/ai2_arc/tree/210d026faf9955653af8916fad021475a3f00453) |
| CUDA | NVIDIA. Toolkit, driver and base-image terms apply separately. [CUDA license](https://docs.nvidia.com/cuda/eula/index.html) |
| FFmpeg and dav1d | FFmpeg developers (LGPL-2.1-or-later), VideoLAN and the dav1d authors (BSD-2-Clause). NInfer pins the archives and compiles a static decoder with its [documented configuration](https://github.com/iamwavecut/ninfer-all/blob/395ce8b85627bcb2d53a1a81623088eab755443c/cmake/NinferFFmpeg.cmake). |

The locally built runtime image includes the full NInfer and media source archives, patches, build recipe, component licenses and notices in `/usr/share/doc/inkling`. This also preserves the notices for GGML, llama-jinja, cpp-httplib, nlohmann/json, spdlog/fmt, utf8proc, XGrammar, DLPack and picojson. The preparation image retains the GGUF and NInfer licenses alongside the Python code. Installed OS and Python packages keep their own license files.

To rebuild against a modified FFmpeg, extract the NInfer and media archives from the runtime image, apply the project patch, update the matching archive and digest in `cmake/MediaSources.cmake`, and rebuild the executable following `scripts/configure.sh` and the Dockerfile's build stage. This procedure allows replacement of the statically linked decoder. The repository publishes source and build instructions. It does not publish compiled images or converted models.
