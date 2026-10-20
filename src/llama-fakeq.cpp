#include "llama-fakeq.h"

#include "llama-impl.h"

#include "ggml.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

enum fakeq_fmt { FQ_NONE, FQ_Q4_0, FQ_MXFP4, FQ_MXFP4B, FQ_NVFP4, FQ_MXFP8,
                 FQ_INT4U, FQ_INT4_E4M3, FQ_INT4_E4M3_G16, FQ_SCALE_E4M3 };

static fakeq_fmt fakeq_parse(const char * name, const char * s) {
    if (!s || !*s) {
        return FQ_NONE;
    }
    const std::string v = s;
    if (v == "q4_0")   { return FQ_Q4_0; }
    if (v == "mxfp4")  { return FQ_MXFP4; }
    if (v == "mxfp4b") { return FQ_MXFP4B; }
    if (v == "nvfp4")  { return FQ_NVFP4; }
    if (v == "mxfp8")  { return FQ_MXFP8; }
    if (v == "int4u")          { return FQ_INT4U; }          // int4, unsigned FP16 scale per 32
    if (v == "int4_e4m3")      { return FQ_INT4_E4M3; }      // int4, UE4M3 scale per 32 + FP32 global
    if (v == "int4_e4m3_g16")  { return FQ_INT4_E4M3_G16; }  // same, per 16
    if (v == "scale_e4m3")     { return FQ_SCALE_E4M3; }     // Q4_0/Q4_1 tensors: block scale d -> E4M3
    GGML_ABORT("%s: unknown format '%s'", name, s);
}

static fakeq_fmt fakeq_w() {
    static const fakeq_fmt f = fakeq_parse("LLAMA_FAKEQ_W", getenv("LLAMA_FAKEQ_W"));
    return f;
}

static fakeq_fmt fakeq_a() {
    static const fakeq_fmt f = fakeq_parse("LLAMA_FAKEQ_A", getenv("LLAMA_FAKEQ_A"));
    return f;
}

// --- element formats -------------------------------------------------------------------------

// nearest E2M1 value (max 6), sign preserved
static inline float e2m1(float v) {
    const float a = fabsf(v);
    float q;
    if      (a < 0.25f) { q = 0.0f; }
    else if (a < 0.75f) { q = 0.5f; }
    else if (a < 1.25f) { q = 1.0f; }
    else if (a < 1.75f) { q = 1.5f; }
    else if (a < 2.5f)  { q = 2.0f; }
    else if (a < 3.5f)  { q = 3.0f; }
    else if (a < 5.0f)  { q = 4.0f; }
    else                { q = 6.0f; }
    return copysignf(q, v);
}

// nearest E4M3FN value, saturating at 448, subnormals down to 2^-9
static inline float e4m3(float v) {
    const float a = fabsf(v);
    if (a == 0.0f) {
        return 0.0f;
    }
    if (a >= 448.0f) {
        return copysignf(448.0f, v);
    }
    const int   e    = std::max((int) floorf(log2f(a)), -6);
    const float step = ldexpf(1.0f, e - 3);
    const float q    = std::min(nearbyintf(a / step) * step, 448.0f);
    return copysignf(q, v);
}

static inline float amax_of(const float * x, int n) {
    float m = 0.0f;
    for (int i = 0; i < n; i++) {
        m = std::max(m, fabsf(x[i]));
    }
    return m;
}

// --- block formats (in place on a float row) -------------------------------------------------

static void fq_q4_0(float * x, int64_t n) {
    for (int64_t b = 0; b < n; b += 32) {
        float * y = x + b;
        float amax = 0.0f, mx = 0.0f;
        for (int j = 0; j < 32; j++) {
            if (amax < fabsf(y[j])) { amax = fabsf(y[j]); mx = y[j]; }
        }
        const float d  = ggml_fp16_to_fp32(ggml_fp32_to_fp16(mx / -8));
        const float id = d ? 1.0f / d : 0.0f;
        for (int j = 0; j < 32; j++) {
            const int q = std::min(15, (int) (int8_t) (y[j] * id + 8.5f));
            y[j] = (q - 8) * d;
        }
    }
}

// MX: shared exponent floor(log2 amax) - emax_elem, optionally also try +1 and keep the better
static void fq_mx(float * x, int64_t n, bool fp8, bool search) {
    const int emax = fp8 ? 8 : 2;
    float tmp[2][32];
    for (int64_t b = 0; b < n; b += 32) {
        float * y = x + b;
        const float amax = amax_of(y, 32);
        if (amax == 0.0f) {
            continue;
        }
        const int e0 = std::clamp((int) floorf(log2f(amax)) - emax, -127, 127);
        float best_err = INFINITY;
        int   best     = 0;
        for (int k = 0; k < (search ? 2 : 1); k++) {
            const float s = ldexpf(1.0f, std::min(e0 + k, 127));
            float err = 0.0f;
            for (int j = 0; j < 32; j++) {
                const float q = (fp8 ? e4m3(y[j] / s) : e2m1(y[j] / s)) * s;
                tmp[k][j] = q;
                err += (q - y[j]) * (q - y[j]);
            }
            if (err < best_err) { best_err = err; best = k; }
        }
        memcpy(y, tmp[best], sizeof(tmp[best]));
    }
}

// NVFP4: 16-block UE4M3 scale relative to an FP32 global scale g = amax/(6*448)
static void fq_nvfp4(float * x, int64_t n, float g) {
    if (g == 0.0f) {
        return;
    }
    for (int64_t b = 0; b < n; b += 16) {
        float * y = x + b;
        const float s = e4m3(amax_of(y, 16) / (6.0f * g)) * g;
        for (int j = 0; j < 16; j++) {
            y[j] = s ? e2m1(y[j] / s) * s : 0.0f;
        }
    }
}

// INT4 (-8..7) with an unsigned block scale: the largest-magnitude element maps to -8 if negative,
// +7 if positive. The scale is FP16, or UE4M3 relative to an FP32 global g (amax / (7 * 448)).
static void fq_int4u(float * x, int64_t n, int bs, bool e4m3_scale, float g) {
    for (int64_t b = 0; b < n; b += bs) {
        float * y = x + b;
        float mx = 0.0f;
        for (int j = 0; j < bs; j++) {
            if (fabsf(mx) < fabsf(y[j])) { mx = y[j]; }
        }
        float d = mx < 0 ? -mx / 8 : mx / 7;
        d = e4m3_scale ? (g ? e4m3(d / g) * g : 0.0f) : ggml_fp16_to_fp32(ggml_fp32_to_fp16(d));
        const float id = d ? 1.0f / d : 0.0f;
        for (int j = 0; j < bs; j++) {
            y[j] = std::clamp(nearbyintf(y[j] * id), -8.0f, 7.0f) * d;
        }
    }
}

static void fq_row(fakeq_fmt f, float * x, int64_t n, float g) {
    switch (f) {
        case FQ_Q4_0:   fq_q4_0(x, n); break;
        case FQ_MXFP4:  fq_mx(x, n, false, false); break;
        case FQ_MXFP4B: fq_mx(x, n, false, true);  break;
        case FQ_MXFP8:  fq_mx(x, n, true,  false); break;
        case FQ_NVFP4:  fq_nvfp4(x, n, g); break;
        case FQ_INT4U:         fq_int4u(x, n, 32, false, 0.0f); break;
        case FQ_INT4_E4M3:     fq_int4u(x, n, 32, true, g * 6.0f / 7.0f); break;
        case FQ_INT4_E4M3_G16: fq_int4u(x, n, 16, true, g * 6.0f / 7.0f); break;
        default: break;
    }
}

// --- weights ---------------------------------------------------------------------------------

static std::mutex fakeq_mtx;
static int    fakeq_n_tensors = 0;
static double fakeq_err = 0.0, fakeq_ref = 0.0;

bool llama_fakeq_weight_enabled() {
    return fakeq_w() != FQ_NONE;
}

void llama_fakeq_weight(ggml_tensor * t) {
    const fakeq_fmt f = fakeq_w();
    if (f == FQ_NONE) {
        return;
    }
    const char * pat = getenv("LLAMA_FAKEQ_W_PAT");
    if (!strstr(ggml_get_name(t), pat && *pat ? pat : "_exps.weight")) {
        return;
    }
    if (f == FQ_SCALE_E4M3) {
        // existing Q4_0 / Q4_1 weights: keep the int4 codes, round each block scale d to E4M3
        // relative to a per-matrix FP32 global (max |d| / 448); the sign of d is kept and Q4_1's min stays FP16
        if (t->type != GGML_TYPE_Q4_0 && t->type != GGML_TYPE_Q4_1) {   // e.g. the Q8_0 MTP layer
            LLAMA_LOG_WARN("%s: skip %s (type %s)\n", __func__, ggml_get_name(t), ggml_type_name(t->type));
            return;
        }
        const size_t bsz   = t->type == GGML_TYPE_Q4_0 ? 18 : 20;
        const int64_t nblk = t->ne[0] / 32 * t->ne[1];   // blocks per matrix
        const int64_t nmat = ggml_nrows(t) / t->ne[1];
        double err = 0.0, ref = 0.0;
        for (int64_t e = 0; e < nmat; e++) {
            uint8_t * base = (uint8_t *) t->data + (size_t) e * nblk * bsz;
            float dmax = 0.0f;
            for (int64_t i = 0; i < nblk; i++) {
                dmax = std::max(dmax, fabsf(ggml_fp16_to_fp32(*(ggml_fp16_t *) (base + i * bsz))));
            }
            const float g = dmax / 448.0f;
            for (int64_t i = 0; i < nblk; i++) {
                ggml_fp16_t * pd = (ggml_fp16_t *) (base + i * bsz);
                const float d  = ggml_fp16_to_fp32(*pd);
                const float dq = g ? e4m3(d / g) * g : 0.0f;
                err += (double) (dq - d) * (dq - d);
                ref += (double) d * d;
                *pd = ggml_fp32_to_fp16(dq);
            }
        }
        std::lock_guard<std::mutex> lk(fakeq_mtx);
        fakeq_n_tensors++;
        fakeq_err += err;   // NMSE of the block scales, not of the weights
        LLAMA_LOG_WARN("%s: scale_e4m3 #%d %s (%s), scale NMSE %.3e\n", __func__, fakeq_n_tensors,
                ggml_get_name(t), ggml_type_name(t->type), ref > 0 ? err / ref : 0.0);
        fakeq_ref += ref;
        return;
    }
    if (t->type != GGML_TYPE_BF16 && t->type != GGML_TYPE_F32) {
        LLAMA_LOG_WARN("%s: skip %s (type %s)\n", __func__, ggml_get_name(t), ggml_type_name(t->type));
        return;
    }
    GGML_ASSERT(ggml_is_contiguous(t));
    const int64_t ne0  = t->ne[0];
    const int64_t rows = t->ne[1];                         // rows per matrix (expert)
    const int64_t nmat = ggml_nrows(t) / rows;
    GGML_ASSERT(ne0 % 32 == 0);

    std::atomic<int64_t> next{0};
    double err_sum = 0.0, ref_sum = 0.0;
    std::mutex acc_mtx;
    auto worker = [&]() {
        std::vector<float> m(ne0 * rows), o(ne0 * rows);
        double err = 0.0, ref = 0.0;
        for (int64_t e; (e = next++) < nmat; ) {
            const size_t off = (size_t) e * ne0 * rows;
            if (t->type == GGML_TYPE_BF16) {
                ggml_bf16_to_fp32_row((const ggml_bf16_t *) t->data + off, m.data(), ne0 * rows);
            } else {
                memcpy(m.data(), (const float *) t->data + off, ne0 * rows * sizeof(float));
            }
            o = m;
            const float g = amax_of(m.data(), (int) (ne0 * rows)) / (6.0f * 448.0f);
            for (int64_t r = 0; r < rows; r++) {
                fq_row(f, m.data() + r * ne0, ne0, g);
            }
            for (int64_t i = 0; i < ne0 * rows; i++) {
                err += (double) (m[i] - o[i]) * (m[i] - o[i]);
                ref += (double) o[i] * o[i];
            }
            if (t->type == GGML_TYPE_BF16) {
                ggml_fp32_to_bf16_row(m.data(), (ggml_bf16_t *) t->data + off, ne0 * rows);
            } else {
                memcpy((float *) t->data + off, m.data(), ne0 * rows * sizeof(float));
            }
        }
        std::lock_guard<std::mutex> lk(acc_mtx);
        err_sum += err;
        ref_sum += ref;
    };
    const int nth = (int) std::min<int64_t>(nmat, std::max(1u, std::thread::hardware_concurrency() / 2));
    std::vector<std::thread> th;
    for (int i = 0; i < nth; i++) {
        th.emplace_back(worker);
    }
    for (auto & x : th) {
        x.join();
    }
    std::lock_guard<std::mutex> lk(fakeq_mtx);
    fakeq_n_tensors++;
    fakeq_err += err_sum;
    fakeq_ref += ref_sum;
}

void llama_fakeq_print_summary() {
    if (fakeq_w() != FQ_NONE) {
        LLAMA_LOG_WARN("%s: weights LLAMA_FAKEQ_W=%s: %d tensors, NMSE = %.4e\n", __func__,
                getenv("LLAMA_FAKEQ_W"), fakeq_n_tensors, fakeq_ref > 0 ? fakeq_err / fakeq_ref : 0.0);
    }
    if (fakeq_a() != FQ_NONE) {
        LLAMA_LOG_WARN("%s: activations LLAMA_FAKEQ_A=%s on expert matmul inputs\n", __func__, getenv("LLAMA_FAKEQ_A"));
    }
}

// --- activations -----------------------------------------------------------------------------

static void fakeq_act_op(ggml_tensor * dst, const ggml_tensor * a, int ith, int nth, void * userdata) {
    const fakeq_fmt f   = (fakeq_fmt) (intptr_t) userdata;
    const int64_t   ne0 = a->ne[0];
    const int64_t   nr  = ggml_nrows(a);
    GGML_ASSERT(a->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 && ne0 % 32 == 0);
    for (int64_t ir = ith; ir < nr; ir += nth) {
        const int64_t i1 = ir % a->ne[1], i2 = (ir / a->ne[1]) % a->ne[2], i3 = ir / (a->ne[1] * a->ne[2]);
        const float * src = (const float *) ((const char *) a->data + i1 * a->nb[1] + i2 * a->nb[2] + i3 * a->nb[3]);
        float * y = (float *) ((char *) dst->data + i1 * dst->nb[1] + i2 * dst->nb[2] + i3 * dst->nb[3]);
        memcpy(y, src, ne0 * sizeof(float));
        fq_row(f, y, ne0, amax_of(y, (int) ne0) / (6.0f * 448.0f));
    }
}

ggml_tensor * llama_fakeq_act(ggml_context * ctx, ggml_tensor * cur) {
    const fakeq_fmt f = fakeq_a();
    if (f == FQ_NONE) {
        return cur;
    }
    GGML_ASSERT(f == FQ_MXFP4 || f == FQ_NVFP4 || f == FQ_MXFP8);
    return ggml_map_custom1(ctx, cur, fakeq_act_op, GGML_N_TASKS_MAX, (void *) (intptr_t) f);
}
