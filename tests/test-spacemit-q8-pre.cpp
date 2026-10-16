// Usage: test-spacemit-q8-pre [reps=200] [cpu=8]
//
// Correctness + speed check for the pre-arranged Q8_0 kernels (GGML_SPACEMIT_Q8_PRE) against the original
// gemm_kernel_i8i8. Standalone (not part of the CMake build); on a K3 board, from the source root:
//
//   G=ggml; S=$G/src/ggml-cpu/spacemit; SPERT=<spine-runtime dir>; B=build-k3
//   g++ -O2 -std=c++17 -fopenmp -DRISCV64_SPACEMIT_IME2 -march=rv64gcv_zfh_zvfh_zba_zicbop_xsmtvdotii -mabi=lp64d \
//     -I$G/include -I$G/src -I$G/src/ggml-cpu -I$S tests/test-spacemit-q8-pre.cpp \
//     $(find $B/ggml/src/CMakeFiles/ggml-cpu.dir -name '*.o') -L$B/bin -lggml-base -L$SPERT/lib -lspert \
//     -Wl,-rpath,$PWD/$B/bin:$SPERT/lib -o test-spacemit-q8-pre
//
// Checks per shape (all must be bit-identical: same instructions on the same values, only the layout differs):
//   - quantize_a_4row_i8_al writes exactly the bytes of quantize_a_4row_i8, only moved to the 192B layout
//   - m4: gemm_kernel_i8i8_pre (152B A) and gemm_kernel_i8i8_pre_al (192B A) on pre-arranged B vs gemm_kernel_i8i8
//   - m1: gemm_kernel_i8i8_pre vs gemm_kernel_i8i8

#include <cstdint>

#include "rvv_kernels.h"
#include "ime_kernels.h"

#include <fcntl.h>
#include <sched.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <vector>

using namespace spacemit_kernels;

static uint32_t g_seed = 1234567;

static uint32_t rnd() {
    g_seed = g_seed * 1664525u + 1013904223u;
    return g_seed >> 8;
}

static float frand() {
    return (rnd() & 0xffff) / 32768.0f - 1.0f;
}

static double now_s() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// same permutation as q8_0x32_prearrange in repack.cpp
static void prearrange(uint8_t * blk_qs) {
    for (int p = 0; p < 4; p++) {
        uint8_t * pair = blk_qs + p * 256;
        uint8_t   tmp[256];
        for (int o = 0; o < 16; o++) {
            memcpy(tmp + o * 16, pair + ((o % 8) * 2 + o / 8) * 16, 16);
        }
        memcpy(pair, tmp, sizeof(tmp));
    }
}

static void * alloc128(size_t n) {
    void * p = aligned_alloc(128, (n + 127) & ~(size_t) 127);
    memset(p, 0, (n + 127) & ~(size_t) 127);
    return p;
}

static int run_shape(int64_t N, int64_t K, int reps) {
    const int64_t kb       = K / 32;
    const size_t  blk_b    = 64 + 1024;  // 32 fp16 scales + 32x32 int8
    const size_t  b_size   = (N / 32) * kb * blk_b;
    uint8_t *     b_orig   = (uint8_t *) alloc128(b_size);
    uint8_t *     b_pre    = (uint8_t *) alloc128(b_size);
    for (int64_t i = 0; i < (N / 32) * kb; i++) {
        _Float16 * d = (_Float16 *) (b_orig + i * blk_b);
        for (int j = 0; j < 32; j++) {
            d[j] = (_Float16) (0.005f + 0.01f * (rnd() & 0xff) / 256.0f);
        }
        for (int j = 0; j < 1024; j++) {
            b_orig[i * blk_b + 64 + j] = (uint8_t) rnd();
        }
    }
    memcpy(b_pre, b_orig, b_size);
    for (int64_t i = 0; i < (N / 32) * kb; i++) {
        prearrange(b_pre + i * blk_b + 64);
    }

    std::vector<float> a(4 * K);
    for (auto & v : a) {
        v = frand();
    }
    const size_t a152_size = kb * 152, a192_size = kb * q8_m4al_blk_size, a1_size = kb * q8_blk_size(32, true);
    uint8_t *    a152      = (uint8_t *) alloc128(a152_size);
    uint8_t *    a192      = (uint8_t *) alloc128(a192_size);
    uint8_t *    a1        = (uint8_t *) alloc128(a1_size);
    rvv::quantize_a_4row_i8(32, a.data(), K, a152);
    rvv::quantize_a_4row_i8_al(32, a.data(), K, a192);
    rvv::quantize_a_row_i8(32, a.data(), K, a1);

    int fails = 0;
    for (int64_t b = 0; b < kb; b++) {
        if (memcmp(a152 + b * 152, a192 + b * 192, 24) != 0 || memcmp(a152 + b * 152 + 24, a192 + b * 192 + 64, 128) != 0) {
            printf("  FAIL quantize_a_4row_i8_al differs at K32 block %lld\n", (long long) b);
            fails++;
            break;
        }
    }

    // one 32-column tile per call, as forward_mul_mat does (the original m4 kernel only handles count_n <= 32:
    // it counts the input-only k_blks operand down to 0 in the first tile)
    using kernel_t = size_t (*)(size_t, const uint8_t *, const uint8_t *, const uint8_t *, float *, size_t, size_t,
                                size_t, size_t);
    auto gemm = [&](kernel_t k, const uint8_t * qa, const uint8_t * qb, float * c, size_t m) {
        for (int64_t ni = 0; ni < N; ni += 32) {
            k(32, qa, qb + ni * kb * (32 + 2), nullptr, c + ni, m, 32, kb, N);
        }
    };
    std::vector<float> c_ref(4 * N), c_pre(4 * N), c_al(4 * N), c1_ref(N), c1_pre(N);
    gemm(ime2::gemm_kernel_i8i8, a152, b_orig, c_ref.data(), 4);
    gemm(ime2::gemm_kernel_i8i8_pre, a152, b_pre, c_pre.data(), 4);
    gemm(ime2::gemm_kernel_i8i8_pre_al, a192, b_pre, c_al.data(), 4);
    gemm(ime2::gemm_kernel_i8i8, a1, b_orig, c1_ref.data(), 1);
    gemm(ime2::gemm_kernel_i8i8_pre, a1, b_pre, c1_pre.data(), 1);

    // the pre-arranged kernels also take a whole row of tiles in one call
    std::vector<float> c_wide(4 * N), c1_wide(N), c1_wide_ref(N);
    ime2::gemm_kernel_i8i8_pre_al(32, a192, b_pre, nullptr, c_wide.data(), 4, N, kb, N);
    // decode: forward_mul_mat's direct 1-row path passes 64 columns per call
    ime2::gemm_kernel_i8i8_pre(32, a1, b_pre, nullptr, c1_wide.data(), 1, N, kb, N);
    ime2::gemm_kernel_i8i8(32, a1, b_orig, nullptr, c1_wide_ref.data(), 1, N, kb, N);

    auto check = [&](const char * name, const std::vector<float> & x, const std::vector<float> & ref) {
        size_t bad = 0;
        for (size_t i = 0; i < ref.size(); i++) {
            bad += memcmp(&x[i], &ref[i], sizeof(float)) != 0;
        }
        printf("  %-28s %s (%zu / %zu differ, ref[0] = %g)\n", name, bad ? "FAIL" : "ok, bit-identical", bad,
               ref.size(), ref[0]);
        fails += bad != 0;
    };
    check("m4 pre (152B A)", c_pre, c_ref);
    check("m4 pre_al (192B A)", c_al, c_ref);
    check("m1 pre", c1_pre, c1_ref);
    check("m4 pre_al, all tiles in one call", c_wide, c_ref);
    check("m1 ref, all tiles in one call", c1_wide_ref, c1_ref);
    check("m1 pre, all tiles in one call", c1_wide, c1_ref);

    // plain fp32 check of the reference itself (row 0, first and last tile) from the dequantized values
    {
        double max_err = 0, max_ref = 0;
        for (int64_t n : { (int64_t) 0, N - 1 }) {
            const int64_t tile = n / 32, col = n % 32;
            double        acc  = 0;
            for (int64_t b = 0; b < kb; b++) {
                const uint8_t * blk = b_orig + (tile * kb + b) * blk_b;
                const float     db  = (float) ((const _Float16 *) blk)[col];
                const float     da  = *(const float *) (a152 + b * 152);
                const int8_t *  qa  = (const int8_t *) (a152 + b * 152 + 24);
                // original layout: column col of the K32 block is the 32 bytes at col * 32
                const int8_t *  qb  = (const int8_t *) (blk + 64 + col * 32);
                int32_t         s   = 0;
                for (int k = 0; k < 32; k++) {
                    s += qa[k] * qb[k];
                }
                acc += (double) da * db * s;
            }
            max_err = std::max(max_err, std::fabs(acc - c_ref[n]));
            max_ref = std::max(max_ref, std::fabs(acc));
        }
        printf("  %-28s max |err| %.3g (|ref| %.3g)\n", "ref kernel vs fp64 (row 0)", max_err, max_ref);
    }

    auto cyc = [&](auto && f) {  // per 32-col x K32 block
        f();
        const double t0 = now_s();
        for (int r = 0; r < reps; r++) {
            f();
        }
        return (now_s() - t0) / reps * 1.8e9 / ((N / 32.0) * kb);
    };
    const double t_ref  = cyc([&] { gemm(ime2::gemm_kernel_i8i8, a152, b_orig, c_ref.data(), 4); });
    const double t_pre  = cyc([&] { gemm(ime2::gemm_kernel_i8i8_pre, a152, b_pre, c_pre.data(), 4); });
    const double t_al   = cyc([&] { gemm(ime2::gemm_kernel_i8i8_pre_al, a192, b_pre, c_al.data(), 4); });
    const double t1_ref = cyc([&] { gemm(ime2::gemm_kernel_i8i8, a1, b_orig, c1_ref.data(), 1); });
    const double t1_pre = cyc([&] { gemm(ime2::gemm_kernel_i8i8_pre, a1, b_pre, c1_pre.data(), 1); });
    printf("  cycles per 32-col x K32 block: m4 %.1f -> pre %.1f -> pre_al %.1f; m1 %.1f -> pre %.1f\n", t_ref, t_pre,
           t_al, t1_ref, t1_pre);

    free(b_orig);
    free(b_pre);
    free(a152);
    free(a192);
    free(a1);
    return fails;
}

int main(int argc, char ** argv) {
    const int reps = argc > 1 ? atoi(argv[1]) : 200;
    const int cpu  = argc > 2 ? atoi(argv[2]) : 8;
    int       fd   = open("/proc/set_ai_thread", O_WRONLY);  // A100 cores only take threads marked as AI threads
    if (fd >= 0) {
        (void) !write(fd, "0", 1);
        close(fd);
    }
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    if (sched_setaffinity(0, sizeof(set), &set) != 0) {
        perror("sched_setaffinity");
    }

    // {N, K}: projection-sized shapes plus a small one
    const int64_t shapes[][2] = { { 512, 2048 }, { 8192, 2048 }, { 2048, 4096 }, { 64, 512 } };
    int           fails       = 0;
    for (auto & s : shapes) {
        printf("N=%lld K=%lld\n", (long long) s[0], (long long) s[1]);
        fails += run_shape(s[0], s[1], reps);
    }
    printf(fails ? "FAILED (%d)\n" : "all ok\n", fails);
    return fails != 0;
}
