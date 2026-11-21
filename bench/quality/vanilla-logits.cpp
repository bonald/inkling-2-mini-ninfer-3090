#include "llama.h"
#include "ggml-backend.h"
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "Usage: vanilla-logits MODEL WINDOWS.ids OUTPUT.f32 PUBLISHER_LIBRARY_DIR\n";
        return 2;
    }
    std::ifstream inputs(argv[2], std::ios::binary);
    std::ofstream output(argv[3], std::ios::binary | std::ios::trunc);
    if (!inputs || !output) return 3;
    ggml_backend_load_all_from_path(argv[4]);
    llama_backend_init();
    auto mp            = llama_model_default_params();
    mp.n_gpu_layers    = 99;
    llama_model* model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 4;
    auto cp            = llama_context_default_params();
    cp.n_ctx           = 512;
    cp.n_seq_max       = 1;
    cp.n_batch         = 64;
    cp.n_ubatch        = 64;
    cp.n_threads       = 4;
    cp.n_threads_batch = 4;
    llama_context* ctx = llama_init_from_model(model, cp);
    if (!ctx) return 5;
    const int32_t voc = llama_vocab_n_tokens(llama_model_get_vocab(model));
    output.write(reinterpret_cast<const char*>(&voc), 4);
    auto batch = llama_batch_init(64, 0, 1);
    std::vector<int32_t> ids(512);
    int windows = 0;
    while (inputs.read(reinterpret_cast<char*>(ids.data()), ids.size() * 4)) {
        llama_memory_clear(llama_get_memory(ctx), true);
        for (int pos = 0; pos < 511; pos += 64) {
            batch.n_tokens = std::min(64, 511 - pos);
            for (int j = 0; j < batch.n_tokens; ++j) {
                batch.token[j]     = ids[pos + j];
                batch.pos[j]       = pos + j;
                batch.n_seq_id[j]  = 1;
                batch.seq_id[j][0] = 0;
                batch.logits[j]    = 1;
            }
            if (llama_decode(ctx, batch)) throw std::runtime_error("llama_decode failed");
            for (int j = 0; j < batch.n_tokens; ++j) {
                const float* row = llama_get_logits_ith(ctx, j);
                if (!row) throw std::runtime_error("missing logits");
                output.write(reinterpret_cast<const char*>(&ids[pos + j + 1]), 4);
                output.write(reinterpret_cast<const char*>(row), std::int64_t(voc) * 4);
            }
        }
        if (!output) throw std::runtime_error("logit write failed");
        ++windows;
    }
    std::cout << windows << " independent windows, " << voc << " output logits per token"
              << std::endl;
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return windows == 8 && inputs.gcount() == 0 ? 0 : 6;
}
