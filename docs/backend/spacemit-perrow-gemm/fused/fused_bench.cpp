// Time the fused GR00T LLM prefill (groot_fused::Decoder, 16 layers, PackedFullI8)
// on this board, outside llama-groot-vla. Weights are synthetic but in the exact
// layerNN.bin layout, so the work and the memory traffic are the real ones.
// usage: fused_bench <weights dir> [tokens=156] [threads=8] [repeats=10]
#include "ggml.h"
#include "llm_fused.inc"

#include <fstream>

static double now_ms() {
    timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e3 + t.tv_nsec * 1e-6;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <weights dir> [tokens] [threads] [repeats]\n", argv[0]);
        return 1;
    }
    const int s       = argc > 2 ? atoi(argv[2]) : 156;
    const int threads = argc > 3 ? atoi(argv[3]) : 8;
    const int repeats = argc > 4 ? atoi(argv[4]) : 10;

    // Same setup as llama-groot-vla's worker thread: AI flag, then the A100 cluster.
    {
        std::ofstream flag("/proc/set_ai_thread");
        flag << "0\n";
    }
    cpu_set_t set;
    CPU_ZERO(&set);
    for (int c = 8; c < 16; ++c) CPU_SET(c, &set);
    for (int i = 0; i < 200 && sched_setaffinity(0, sizeof(set), &set) != 0; ++i) usleep(1000);

    ggml_init_params ip = { 64 * 1024 * 1024, nullptr, false };
    ggml_context *   ctx = ggml_init(ip);
    ggml_tensor *    big = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8192, s);  // hidden + 3 deepstack taps
    ggml_tensor *    in  = ggml_view_2d(ctx, big, 2048, s, big->nb[1], 0);
    ggml_tensor *    pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 3 * s);
    ggml_tensor *    dst = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, s);
    srand(1);
    for (int64_t i = 0; i < ggml_nelements(big); ++i) ((float *) big->data)[i] = (rand() % 2001 - 1000) * 1e-3f;
    for (int d = 0; d < 3; ++d)
        for (int r = 0; r < s; ++r) ((int32_t *) pos->data)[d * s + r] = r;

    groot_fused::Decoder dec(argv[1], threads, 512);
    std::vector<double>  t;
    for (int i = 0; i < repeats + 1; ++i) {
        double t0 = now_ms();
        groot_fused::Decoder::callback(dst, in, pos, 0, 1, &dec);
        double t1 = now_ms();
        if (!dec.error.empty()) {
            fprintf(stderr, "decoder error: %s\n", dec.error.c_str());
            return 1;
        }
        if (i > 0) t.push_back(t1 - t0);
    }
    std::sort(t.begin(), t.end());
    double sum = 0;
    for (double v : t) sum += v;
    printf("fused 16L prefill s=%d threads=%d: best %.2f median %.2f mean %.2f ms (%d runs)\n", s, threads, t.front(),
           t[t.size() / 2], sum / t.size(), (int) t.size());
    printf("out[0..3] %.4f %.4f %.4f %.4f\n", ((float *) dst->data)[0], ((float *) dst->data)[1],
           ((float *) dst->data)[2], ((float *) dst->data)[3]);
    ggml_free(ctx);
    return 0;
}
