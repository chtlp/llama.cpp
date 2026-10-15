// Usage: test-spacemit-hp-aligned [reps=200] [cpu=8]
//
// Correctness + speed check for the aligned-A / signed-B Q4_0 hp kernels (GGML_SPACEMIT_HP_ALIGNED) against the
// original hp kernels and an FP32 reference. Standalone (not part of the CMake build); on a K3 board:
//
//   G=ggml; S=$G/src/ggml-cpu/spacemit; SPERT=<spine-runtime dir>
//   g++ -O2 -std=c++17 -fopenmp -DRISCV64_SPACEMIT_IME2 -march=rv64gcv_zfh_zvfh_zba_zicbop_xsmtvdotii -mabi=lp64d \
//     -I$G/include -I$G/src -I$G/src/ggml-cpu -I$S tests/test-spacemit-hp-aligned.cpp \
//     $(find build/ggml/src/CMakeFiles/ggml-cpu.dir -name '*.o') -Lbuild/bin -lggml-base -L$SPERT/lib -lspert \
//     -Wl,-rpath,$PWD/build/bin:$SPERT/lib -o test-spacemit-hp-aligned
//
// (linked against the ggml-cpu objects: libggml-cpu.so is built with -fvisibility=hidden)
//
// Checks per shape:
//   - quantize_a_4row_i8_hp_al writes exactly the bytes of quantize_a_4row_i8_hp, only rearranged
//   - m4: new kernel vs old kernel (expected ~1e-3 relative, fp16 accumulation order) and both vs FP32
//   - m1: signed-B kernel vs old kernel
//   - A and B end right before a PROT_NONE page, so any read past either buffer faults

#include <cstdint>

#include "rvv_kernels.h"
#include "ime_kernels.h"

#include <sched.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <vector>

// original hp kernels (defined in ime2_kernels.cpp, not declared in a header)
namespace spacemit_kernels::ime2 {
void gemm_kernel_i8i4_hp_m4(size_t, const uint8_t *, const uint8_t *, const uint8_t *, float *, size_t, size_t,
                            size_t, size_t);
void gemm_kernel_i8i4_hp_m1(size_t, const uint8_t *, const uint8_t *, const uint8_t *, float *, size_t, size_t,
                            size_t, size_t);
}  // namespace spacemit_kernels::ime2

using namespace spacemit_kernels;

static uint32_t g_seed = 1234567;

static uint32_t rnd() {
    g_seed = g_seed * 1664525u + 1013904223u;
    return g_seed >> 8;
}

static float frand() {
    return (rnd() & 0xffff) / 32768.0f - 1.0f;
}

// buffer of `size` bytes ending exactly at a PROT_NONE guard page
struct guarded {
    uint8_t * base = nullptr;
    size_t    map  = 0;
    uint8_t * p    = nullptr;

    explicit guarded(size_t size) {
        const size_t pg = (size_t) sysconf(_SC_PAGESIZE);
        map             = (size + pg - 1) / pg * pg + pg;
        base = (uint8_t *) mmap(nullptr, map, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (base == MAP_FAILED) {
            perror("mmap");
            exit(1);
        }
        mprotect(base + map - pg, pg, PROT_NONE);
        p = base + map - pg - size;
    }

    ~guarded() { munmap(base, map); }
};

static double rel_err(const float * x, const float * ref, size_t n) {
    double num = 0, den = 0;
    for (size_t i = 0; i < n; i++) {
        num += (double) (x[i] - ref[i]) * (x[i] - ref[i]);
        den += (double) ref[i] * ref[i];
    }
    return std::sqrt(num / (den > 0 ? den : 1));
}

static double now_s() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// B in the hp repack layout: per N32 tile, per K256: 8 x [fp16 d[32]][qs 32 x 16B]. Same nibble order as
// make_block_q4_0x32 in repack.cpp: per column, byte j holds elements (2j, 2j+1) for j < 8 and (16+2j', 16+2j'+1)
// for the second half. `q` is the unsigned 4-bit value (0..15), w = d * (q - 8).
static void make_b(size_t n, size_t k, std::vector<uint8_t> & b_old, std::vector<float> & w) {
    const size_t kb = k / 256;
    b_old.assign(n / 32 * kb * 4608, 0);
    w.assign(n * k, 0);
    for (size_t col = 0; col < n; col++) {
        for (size_t s = 0; s < k / 32; s++) {
            const _Float16 d = (_Float16) (0.002f + 0.02f * (rnd() & 0xff) / 255.0f);
            uint8_t        q[32];
            for (int i = 0; i < 32; i++) {
                q[i]                       = rnd() & 0xf;
                w[col * k + s * 32 + i] = (float) d * (q[i] - 8);
            }
            uint8_t * blk = b_old.data() + ((col / 32) * kb + s / 8) * 4608 + (s % 8) * 576;
            ((_Float16 *) blk)[col % 32] = d;
            uint8_t * qs                 = blk + 64 + (col % 32) * 16;
            for (int j = 0; j < 8; j++) {
                qs[j]     = q[2 * j] | (q[2 * j + 1] << 4);
                qs[8 + j] = q[16 + 2 * j] | (q[16 + 2 * j + 1] << 4);
            }
        }
    }
}

static int check(size_t n, size_t k, int reps) {
    const size_t kb = k / 256;
    printf("== N=%zu K=%zu\n", n, k);

    std::vector<float> a(4 * k);
    for (auto & v : a) {
        v = frand() * (1.0f + 3.0f * (rnd() % 7 == 0));
    }
    std::vector<uint8_t> b_old;
    std::vector<float>   w;
    make_b(n, k, b_old, w);

    // FP32 reference
    std::vector<float> ref(4 * n);
    for (int r = 0; r < 4; r++) {
        for (size_t c = 0; c < n; c++) {
            double s = 0;
            for (size_t i = 0; i < k; i++) {
                s += (double) a[r * k + i] * w[c * k + i];
            }
            ref[r * n + c] = (float) s;
        }
    }

    // old path
    std::vector<uint8_t> qa_old(kb * 1160 + 64);
    rvv::quantize_a_4row_i8_hp(256, a.data(), k, qa_old.data());
    std::vector<float> c_old(4 * n), c_new(4 * n), c1_old(n), c1_new(n);
    ime2::gemm_kernel_i8i4_hp_m4(256, qa_old.data(), b_old.data(), nullptr, c_old.data(), 4, n, kb, n);

    // new path: buffers end at a guard page
    guarded qa_new(kb * q8_hp_m4al_blk_size);
    guarded b_new(b_old.size());
    if (((uintptr_t) qa_new.p & 63) || ((uintptr_t) b_new.p & 63)) {
        printf("FAIL buffers not 64B aligned\n");
        return 1;
    }
    for (size_t i = 0; i < b_old.size(); i++) {
        const bool is_qs = (i % 576) >= 64;
        b_new.p[i]       = is_qs ? (b_old[i] ^ 0x88) : b_old[i];
    }
    rvv::quantize_a_4row_i8_hp_al(256, a.data(), k, qa_new.p);

    // layout check: same bytes, only moved (scale headers: only the first fp16 is defined)
    int bad = 0;
    for (size_t s = 0; s < kb; s++) {
        const uint8_t * o  = qa_old.data() + s * 1160;
        const uint8_t * nw = qa_new.p + s * q8_hp_m4al_blk_size;
        for (int kk = 0; kk < 8; kk++) {
            bad += memcmp(nw + kk * 128, o + kk * 136 + 8, 128) != 0;
            bad += memcmp(nw + 1024 + kk * 8, o + kk * 136, 2) != 0;
        }
        bad += memcmp(nw + 1088, o + 1088, 72) != 0;
    }
    printf("  quantize_a_4row_i8_hp_al layout: %s\n", bad ? "MISMATCH" : "bit-identical to quantize_a_4row_i8_hp");

    ime2::gemm_kernel_i8i4_hp_m4_al(256, qa_new.p, b_new.p, nullptr, c_new.data(), 4, n, kb, n);
    std::vector<float> c_disp(4 * n);
    const size_t       rows = ime2::gemm_kernel_i8i4_hp_al(256, qa_new.p, b_new.p, nullptr, c_disp.data(), 4, n, kb, n);
    const bool         disp_ok = rows == 4 && memcmp(c_disp.data(), c_new.data(), c_new.size() * 4) == 0;

    const double e_new_old = rel_err(c_new.data(), c_old.data(), c_new.size());
    const double e_old_ref = rel_err(c_old.data(), ref.data(), ref.size());
    const double e_new_ref = rel_err(c_new.data(), ref.data(), ref.size());
    printf("  m4: new vs old %.2e | old vs fp32 %.2e | new vs fp32 %.2e | dispatcher %s\n", e_new_old, e_old_ref,
           e_new_ref, disp_ok ? "ok" : "MISMATCH");

    // m1 (1-row layout, unchanged) with signed B
    std::vector<uint8_t> qa1(kb * 290 + 64);
    rvv::quantize_a_row_i8_hp(256, a.data(), k, qa1.data());
    ime2::gemm_kernel_i8i4_hp_m1(256, qa1.data(), b_old.data(), nullptr, c1_old.data(), 1, n, kb, n);
    guarded qa1g(kb * 290);
    memcpy(qa1g.p, qa1.data(), kb * 290);
    const size_t rows1 = ime2::gemm_kernel_i8i4_hp_al(256, qa1g.p, b_new.p, nullptr, c1_new.data(), 1, n, kb, n);
    const double e1    = rel_err(c1_new.data(), c1_old.data(), n);
    const double e1ref = rel_err(c1_new.data(), ref.data(), n);
    printf("  m1: signed vs old %.2e | signed vs fp32 %.2e | rows=%zu\n", e1, e1ref, rows1);

    int fail = bad || !disp_ok || rows1 != 1 || e_new_old > 5e-3 || e1 > 5e-3 || e_new_ref > 2 * e_old_ref + 1e-3;

    if (reps > 0) {
        // single-thread speed: cycles per 4x32xK32 subblock at 1.8 GHz
        const double subblks = (double) (n / 32) * (k / 32);
        double       t0      = now_s();
        for (int i = 0; i < reps; i++) {
            ime2::gemm_kernel_i8i4_hp_m4(256, qa_old.data(), b_old.data(), nullptr, c_old.data(), 4, n, kb, n);
        }
        double t1 = now_s();
        for (int i = 0; i < reps; i++) {
            ime2::gemm_kernel_i8i4_hp_m4_al(256, qa_new.p, b_new.p, nullptr, c_new.data(), 4, n, kb, n);
        }
        double t2 = now_s();
        const double cyc_old = (t1 - t0) / reps / subblks * 1.8e9;
        const double cyc_new = (t2 - t1) / reps / subblks * 1.8e9;
        printf("  speed (1 thread, cycles/subblock @1.8GHz): old m4 %.1f, new m4 %.1f (%.2fx)\n", cyc_old, cyc_new,
               cyc_old / cyc_new);
    }

    printf("  %s\n", fail ? "FAIL" : "PASS");
    return fail;
}

// IME instructions need the thread registered as an AI thread and running on an A100 core (CPU 8-15 on K3)
static void setup_ai_thread(int cpu) {
    FILE * f = fopen("/proc/set_ai_thread", "w");
    if (f) {
        fprintf(f, "%ld\n", (long) syscall(SYS_gettid));
        fclose(f);
    } else {
        printf("warning: cannot open /proc/set_ai_thread\n");
    }
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    if (sched_setaffinity(0, sizeof(set), &set) != 0) {
        printf("warning: cannot pin to CPU %d\n", cpu);
    }
}

int main(int argc, char ** argv) {
    const int reps = argc > 1 ? atoi(argv[1]) : 200;
    setup_ai_thread(argc > 2 ? atoi(argv[2]) : 8);
    int fail = 0;
    fail += check(32, 256, reps);
    fail += check(96, 2048, reps);
    fail += check(256, 768, reps);
    fail += check(512, 2048, reps);
    fail += check(64, 4096, reps);
    printf("%s\n", fail ? "SOME CHECKS FAILED" : "ALL PASS");
    return fail ? 1 : 0;
}
