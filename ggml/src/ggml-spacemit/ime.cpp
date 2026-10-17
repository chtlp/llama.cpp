#define GGML_COMMON_IMPL_CPP
#define GGML_COMMON_DECL_CPP

#include "ime.h"

#include "common.h"
#include "ggml-backend-impl.h"
#include "ggml-common.h"
#include "ggml-cpu.h"
#include "ime_kernels.h"
#include "repack.h"
#include "rvv_kernels.h"
#include "scalar_kernels.h"
#include "spacemit-context.h"
#include "spacemit-env.h"
#include "spine_mem_pool.h"
#include "vec.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cerrno>
#include <cmath>
#include <cstdio>  // for GGML_ASSERT
#include <stdexcept>
#include <thread>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
// clang-format off
#if defined(__riscv)

#if !defined(__riscv_v) || !defined(__riscv_v_intrinsic)
#error "riscv v extension or v_intrinsic not enabled"
#else
#include <riscv_vector.h>
#endif

#if !defined(__riscv_zfh) || !defined(__riscv_zvfh)
#error "riscv zfh extension not enabled, GGML_RV_ZFH and GGML_RV_ZVFH must be defined to 1"
#endif

#if !defined(__riscv_zba)
#error "riscv zba extension not enabled, GGML_RV_ZBA must be defined to 1"
#endif

#if defined(RISCV64_SPACEMIT_IME1) || defined(RISCV64_SPACEMIT_IME2)
#else
#error "RISCV64_SPACEMIT_IME1 or RISCV64_SPACEMIT_IME2 not defined"
#endif

#else

#error "riscv not enabled in this build"

#endif

#if defined(__GNUC__)
#pragma GCC diagnostic ignored "-Woverlength-strings"
#pragma GCC diagnostic ignored "-Wcast-qual"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

// clang-format on

namespace ggml::cpu::riscv64_spacemit {

template <typename BLOC_TYPE, int64_t INTER_SIZE, int64_t NB_COLS> constexpr size_t get_repacked_block_type_size() {
    if constexpr (std::is_same_v<BLOC_TYPE, block_q6_K> || std::is_same_v<BLOC_TYPE, block_q8_0>) {
        return sizeof(block_q8_0);
    } else if constexpr (std::is_same_v<BLOC_TYPE, block_q4_0>) {
        return sizeof(block_q4_0) * INTER_SIZE / QK4_0;
    } else if constexpr (std::is_same_v<BLOC_TYPE, block_q4_1> || std::is_same_v<BLOC_TYPE, block_q4_K>) {
        return (sizeof(block_q4_0) + sizeof(uint8_t)) * INTER_SIZE / QK4_1;
    } else if constexpr (std::is_same_v<BLOC_TYPE, block_q2_K>) {
        return sizeof(spacemit_kernels::nrow_block_q2_k<1>);
    } else if constexpr (std::is_same_v<BLOC_TYPE, block_q3_K>) {
        return sizeof(spacemit_kernels::nrow_block_q3_k<1>);
    } else if constexpr (std::is_same_v<BLOC_TYPE, block_mxfp4>) {
        return sizeof(spacemit_kernels::nrow_block_mxfp4<1>);
    } else if constexpr (std::is_same_v<BLOC_TYPE, block_q5_1> || std::is_same_v<BLOC_TYPE, block_q5_K>) {
        return sizeof(spacemit_kernels::nrow_block_q5_1<1>);
    } else if constexpr (std::is_same_v<BLOC_TYPE, block_q5_0>) {
        return sizeof(spacemit_kernels::nrow_block_q5_0<1>);
    } else {
        assert(false);
        return 0;
    }
}

template <typename BLOC_TYPE> constexpr bool block_type_has_zp() {
    if constexpr (std::is_same_v<BLOC_TYPE, block_q6_K> || std::is_same_v<BLOC_TYPE, block_q8_0> ||
                  std::is_same_v<BLOC_TYPE, block_q3_K> || std::is_same_v<BLOC_TYPE, block_q4_0> ||
                  std::is_same_v<BLOC_TYPE, block_mxfp4> || std::is_same_v<BLOC_TYPE, block_q5_0>) {
        return false;
    } else if constexpr (std::is_same_v<BLOC_TYPE, block_q4_1> || std::is_same_v<BLOC_TYPE, block_q4_K> ||
                         std::is_same_v<BLOC_TYPE, block_q2_K> || std::is_same_v<BLOC_TYPE, block_q5_1> ||
                         std::is_same_v<BLOC_TYPE, block_q5_K>) {
        return true;
    } else {
        assert(false);
        return false;
    }
}

// ---- ported from ggml-cpu/spacemit (linpengt/qwen35-opt): matmul kernels, MoE m8/m4 paths, fusions ----
// GGML_OP_STATS sub-phase timing lives in ggml-cpu (op-stats.c); not available in this backend.
static inline bool    ggml_op_stats_enabled() { return false; }
static inline int64_t ggml_op_stats_now_ns() { return 0; }

// MUL_MAT_ID sub-phase stats, enabled with GGML_OP_STATS=1. Index 0: single-token graphs, 1: multi-token.
struct alignas(64) mmid_phase_stats {
    int64_t calls;
    int64_t t_total;
    int64_t t_quant;
    int64_t t_route;
    int64_t t_sync;
    int64_t t_wcopy;
    int64_t t_agather;
    int64_t t_kernel;
    int64_t rows_m4;
    int64_t rows_m2;
    int64_t rows_m1;
    int64_t t_quant4;
    int64_t w_tcm_hit;
    int64_t w_tcm_miss;
    int64_t fast_path;
    double  kernel_flops;
    // m8/m4 group path, split out of t_kernel / t_quant4
    int64_t t_k8;
    int64_t t_k4;
    int64_t t_scatter;
    int64_t t_q4wait;
    int64_t t_scatter_wide;  // part of t_scatter for outputs wider than 1024 (ffn_down)
    int64_t rows_m8;
    double  flops_m8;
    double  flops_m4;
};

// Dense MUL_MAT (multi-row) sub-phase stats, GGML_OP_STATS=1. Index: 0 q4_0, 1 q8_0, 2 other.
struct alignas(64) dense_phase_stats {
    int64_t calls;
    int64_t t_total;
    int64_t t_quant;
    int64_t t_sync;
    int64_t t_acopy;
    int64_t t_bcopy;
    int64_t t_barrier;
    int64_t t_gemm;
    int64_t branch[3];
    double  flops;
    double  b_bytes;  // weight bytes the kernels read (per pass over B)
};

struct mmid_route_stats {
    int64_t calls;
    int64_t active_experts;
    int64_t rows;
    int64_t hist[9];  // rows per active expert: 1, 2-3, 4-7, ..., 128-255, >=256
    int64_t tcm_size;
};

// Prototype: run MoE expert rows through the 4-row hp kernel (GGML_SPACEMIT_MOE_M4=0 disables).
static bool mmid_m4_enabled() {
    static const bool enabled = [] {
        const char * e = getenv("GGML_SPACEMIT_MOE_M4");
        return !(e && e[0] == '0');
    }();
    return enabled;
}

// Graph-level fusions, decided once per graph compute by ggml_spacemit_graph_prepare()
// and recorded in spare op_params slots of the nodes involved:
//  - conv: concat(conv_states, transpose(qkv)) -> ssm_conv. SSM_CONV reads the two concat inputs directly
//    and CONCAT only writes the trailing columns that are read back through views (the conv state update).
//  - moe_w: mul(mul_mat_id(...), weights). MUL_MAT_ID scales each output row by its routing weight as it
//    writes it, and the MUL becomes a no-op (or a copy when it is not in place).
// GGML_SPACEMIT_FUSE_CONV=0 / GGML_SPACEMIT_FUSE_MOE_W=0 disable them.
static bool fuse_env_enabled(const char * name) {
    const char * e = getenv(name);
    return !(e && e[0] == '0');
}

// RVV replacements for ops that otherwise run as scalar generic code (GGML_SPACEMIT_RVV_RMS_MUL / _SIGMOID / _L2=0
// turn them off): the fused RMS_NORM + MUL, SIGMOID and L2_NORM
static bool rvv_rms_mul_enabled() {
    static const bool v = fuse_env_enabled("GGML_SPACEMIT_RVV_RMS_MUL");
    return v;
}

static bool rvv_sigmoid_enabled() {
    static const bool v = fuse_env_enabled("GGML_SPACEMIT_RVV_SIGMOID");
    return v;
}

static bool rvv_l2_enabled() {
    static const bool v = fuse_env_enabled("GGML_SPACEMIT_RVV_L2");
    return v;
}

// MoE m8/m4 row write-out: scale by the routing weight while copying (GGML_SPACEMIT_MOE_STORE1=0: copy, then scale
// in place as before)
static bool moe_store1_enabled() {
    static const bool v = fuse_env_enabled("GGML_SPACEMIT_MOE_STORE1");
    return v;
}

// MoE: rows left after the full 8-row groups of an expert (1..7, including experts with fewer than 8 rows) run as
// one zero-padded m8 group instead of a 4-row group plus single rows (GGML_SPACEMIT_MOE_PAD8=0 disables)
static bool moe_pad8_enabled() {
    static const bool v = fuse_env_enabled("GGML_SPACEMIT_MOE_PAD8");
    return v;
}

// MoE: the up and gate MUL_MAT_ID of a layer are adjacent and read the same activations with the same expert ids;
// the second one reuses the routing tables and 8-row groups the first one left in the work buffer
// (GGML_SPACEMIT_MOE_REUSEQ=0 disables)
static bool moe_reuseq_enabled() {
    static const bool v = fuse_env_enabled("GGML_SPACEMIT_MOE_REUSEQ");
    return v;
}

static bool fuse_conv_enabled() {
    static const bool enabled = fuse_env_enabled("GGML_SPACEMIT_FUSE_CONV");
    return enabled;
}

static bool fuse_moe_w_enabled() {
    static const bool enabled = fuse_env_enabled("GGML_SPACEMIT_FUSE_MOE_W");
    return enabled;
}

// op_params slots (int32 index); CONCAT uses [0] (dim), MUL_MAT_ID and MUL use none.
constexpr int FUSE_P_PTR     = 8;   // MUL_MAT_ID: weights tensor pointer (2 slots)
constexpr int FUSE_P_USES    = 11;  // scratch during graph prepare
constexpr int FUSE_P_APPLIED = 12;  // MUL_MAT_ID: set to 1 by the kernel once it applied the weights
constexpr int FUSE_P_REUSEQ  = 13;  // MUL_MAT_ID: 1 = reuse routing + 8-row quantization of the previous node
constexpr int FUSE_P_AUX     = 14;  // scratch during graph prepare
constexpr int FUSE_P_STATE   = 15;  // CONCAT: first column to write + 1 (0 = not fused); MUL_MAT_ID: 1 = fused

static void fuse_set_ptr(ggml_tensor * t, const ggml_tensor * p) {
    memcpy((char *) t->op_params + FUSE_P_PTR * sizeof(int32_t), &p, sizeof(p));
}

static const ggml_tensor * fuse_get_ptr(const ggml_tensor * t) {
    const ggml_tensor * p;
    memcpy(&p, (const char *) t->op_params + FUSE_P_PTR * sizeof(int32_t), sizeof(p));
    return p;
}

static bool is_transposed_concat_f32(const ggml_tensor * t) {
    if (t->op != GGML_OP_CONCAT || ggml_get_op_params_i32(t, 0) != 0) {
        return false;
    }
    const ggml_tensor * s0 = t->src[0];
    const ggml_tensor * s1 = t->src[1];
    return t->type == GGML_TYPE_F32 && s0->type == GGML_TYPE_F32 && s1->type == GGML_TYPE_F32 &&
           t->nb[0] == sizeof(float) && s0->nb[0] == sizeof(float) && s1->nb[1] == sizeof(float) &&
           s1->nb[0] != sizeof(float) && s0->ne[1] == t->ne[1] && s1->ne[1] == t->ne[1] && s0->ne[2] == t->ne[2] &&
           s1->ne[2] == t->ne[2] && t->ne[3] == 1 && !(t->flags & GGML_TENSOR_FLAG_OUTPUT);
}

// First column of concat c covered by view v, or -1 if v is not a plain window of whole rows' trailing
// columns (same row/plane strides, starting in row 0).
static int64_t concat_view_first_col(const ggml_tensor * c, const ggml_tensor * v) {
    if (v->type != c->type || v->nb[0] != c->nb[0] || (v->ne[1] > 1 && v->nb[1] != c->nb[1]) ||
        (v->ne[2] > 1 && v->nb[2] != c->nb[2]) || v->ne[3] != 1 || v->view_offs / c->nb[1] != 0) {
        return -1;
    }
    const int64_t col = (int64_t) ((v->view_offs % c->nb[1]) / c->nb[0]);
    if (col + v->ne[0] > c->ne[0] || v->ne[1] > c->ne[1] || v->ne[2] > c->ne[2]) {
        return -1;
    }
    return col;
}

static bool rvv_op_disabled(const ggml_tensor * op);

// Both halves of a fusion must run in this backend's kernels (a generic fallback would read unwritten
// concat columns or apply the routing weights twice), so ops routed elsewhere by GGML_SPACEMIT_RVV_OFF
// are never fused.
static void graph_prepare_fusions(ggml_cgraph * gf) {
    const int n = ggml_graph_n_nodes(gf);

    // reset; mark candidates
    for (int i = 0; i < n; ++i) {
        ggml_tensor * t = ggml_graph_node(gf, i);
        if (t->op == GGML_OP_CONCAT) {
            const bool cand = fuse_conv_enabled() && is_transposed_concat_f32(t) && !rvv_op_disabled(t) &&
                              t->src[0]->ne[0] == 3;
            ggml_set_op_params_i32(t, FUSE_P_STATE, cand ? -1 : 0);  // -1: candidate, still valid
            ggml_set_op_params_i32(t, FUSE_P_AUX, (int32_t) t->ne[0]);  // min column read through views
            ggml_set_op_params_i32(t, FUSE_P_USES, 0);                  // ssm_conv users
        } else if (t->op == GGML_OP_MUL_MAT_ID) {
            ggml_set_op_params_i32(t, FUSE_P_STATE, 0);
            ggml_set_op_params_i32(t, FUSE_P_APPLIED, 0);
            ggml_set_op_params_i32(t, FUSE_P_USES, 0);
            ggml_set_op_params_i32(t, FUSE_P_REUSEQ, 0);
        }
    }

    // adjacent MUL_MAT_IDs with the same activations, expert ids and expert matrix shape/type (ffn_up -> ffn_gate):
    // nothing runs between them, so the work buffer still holds the first one's routing and quantized groups
    if (moe_reuseq_enabled()) {
        for (int i = 0; i + 1 < n; ++i) {
            const ggml_tensor * a = ggml_graph_node(gf, i);
            ggml_tensor *       b = ggml_graph_node(gf, i + 1);
            if (a->op != GGML_OP_MUL_MAT_ID || b->op != GGML_OP_MUL_MAT_ID || a->src[1] != b->src[1] ||
                a->src[2] != b->src[2] || a->src[0]->type != GGML_TYPE_Q4_0 || b->src[0]->type != GGML_TYPE_Q4_0 ||
                !ggml_are_same_shape(a->src[0], b->src[0]) || a->src[0]->extra == nullptr || b->src[0]->extra == nullptr ||
                (b->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 || (a->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
                continue;
            }
            ggml_set_op_params_i32(b, FUSE_P_REUSEQ, 1);
        }
    }

    auto concat_view_use = [](ggml_tensor * c, const ggml_tensor * v) {
        const int64_t col = concat_view_first_col(c, v);
        if (col < 0) {
            ggml_set_op_params_i32(c, FUSE_P_STATE, 0);
        } else if (col < ggml_get_op_params_i32(c, FUSE_P_AUX)) {
            ggml_set_op_params_i32(c, FUSE_P_AUX, (int32_t) col);
        }
    };

    // scan all uses
    for (int i = 0; i < n; ++i) {
        ggml_tensor * t = ggml_graph_node(gf, i);
        if (t->view_src && t->view_src->op == GGML_OP_CONCAT && ggml_get_op_params_i32(t->view_src, FUSE_P_STATE)) {
            concat_view_use(t->view_src, t);
        }
        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            ggml_tensor * s = t->src[j];
            if (s == nullptr) {
                continue;
            }
            if (s->op == GGML_OP_CONCAT && ggml_get_op_params_i32(s, FUSE_P_STATE)) {
                if (t->view_src == s) {
                    // the view node itself, handled above
                } else if (t->op == GGML_OP_SSM_CONV && j == 0 && t->src[1]->ne[0] == 4 && !rvv_op_disabled(t) &&
                           t->src[2] == s->src[1] && t->src[3] == s->src[0]) {
                    // the graph keeps the concat inputs alive until this node (build_ssm_conv)
                    ggml_set_op_params_i32(s, FUSE_P_USES, ggml_get_op_params_i32(s, FUSE_P_USES) + 1);
                } else {
                    ggml_set_op_params_i32(s, FUSE_P_STATE, 0);
                }
            } else if (s->view_src && s->view_src->op == GGML_OP_CONCAT &&
                       ggml_get_op_params_i32(s->view_src, FUSE_P_STATE)) {
                concat_view_use(s->view_src, s);
            }
            const ggml_tensor * m = s->view_src ? s->view_src : s;
            if (m->op == GGML_OP_MUL_MAT_ID) {
                ggml_tensor * mm = const_cast<ggml_tensor *>(m);
                ggml_set_op_params_i32(mm, FUSE_P_USES, ggml_get_op_params_i32(mm, FUSE_P_USES) + 1);
            }
        }
    }

    std::unordered_map<const ggml_tensor *, int> node_idx;
    if (fuse_moe_w_enabled()) {
        node_idx.reserve(n);
        for (int i = 0; i < n; ++i) {
            node_idx[ggml_graph_node(gf, i)] = i;
        }
    }
    auto computed_before = [&](const ggml_tensor * a, const ggml_tensor * b) {
        const ggml_tensor * ab = a->view_src ? a->view_src : a;
        const auto          ia = node_idx.find(a);
        const auto          iab = node_idx.find(ab);
        const auto          ib = node_idx.find(b);
        if (ib == node_idx.end()) {
            return false;
        }
        // a leaf (not a node) is always available; otherwise a and its base must come first
        return (ia == node_idx.end() || ia->second < ib->second) && (iab == node_idx.end() || iab->second < ib->second);
    };

    for (int i = 0; i < n; ++i) {
        ggml_tensor * t = ggml_graph_node(gf, i);
        if (t->op == GGML_OP_CONCAT) {
            const bool ok = ggml_get_op_params_i32(t, FUSE_P_STATE) == -1 && ggml_get_op_params_i32(t, FUSE_P_USES) > 0;
            ggml_set_op_params_i32(t, FUSE_P_STATE, ok ? ggml_get_op_params_i32(t, FUSE_P_AUX) + 1 : 0);
        } else if (t->op == GGML_OP_MUL && fuse_moe_w_enabled() && !rvv_op_disabled(t)) {
            ggml_tensor *       m = t->src[0];
            const ggml_tensor * w = t->src[1];
            if (m->op == GGML_OP_MUL_MAT_ID && m->view_src == nullptr && ggml_get_op_params_i32(m, FUSE_P_USES) == 1 &&
                !(m->flags & GGML_TENSOR_FLAG_OUTPUT) && m->type == GGML_TYPE_F32 && t->type == GGML_TYPE_F32 &&
                w->type == GGML_TYPE_F32 && w->ne[0] == 1 && w->ne[1] == m->ne[1] && w->ne[2] == m->ne[2] &&
                m->ne[3] == 1 && w->ne[3] == 1 && ggml_are_same_shape(t, m) && ggml_is_contiguous(m) &&
                ggml_is_contiguous(t) && computed_before(w, m)) {
                ggml_set_op_params_i32(m, FUSE_P_STATE, 1);
                fuse_set_ptr(m, w);
            }
        }
    }
}

// MUL_MAT_ID side of the moe_w fusion: routing weight for output row (i1 = slot, i2 = token), or nullptr
// when the op is not fused.
static const ggml_tensor * mmid_fused_weights(ggml_tensor * op) {
    if (ggml_get_op_params_i32(op, FUSE_P_STATE) != 1) {
        return nullptr;
    }
    ggml_set_op_params_i32(op, FUSE_P_APPLIED, 1);
    return fuse_get_ptr(op);
}

static inline void mmid_scale_row(float * y, int64_t n, const ggml_tensor * w, int64_t i1, int64_t i2) {
    if (w == nullptr) {
        return;
    }
    const float s = *(const float *) ((const char *) w->data + i1 * w->nb[1] + i2 * w->nb[2]);
    for (int64_t i = 0; i < n;) {
        const size_t vl = __riscv_vsetvl_e32m8(n - i);
        __riscv_vse32_v_f32m8(y + i, __riscv_vfmul_vf_f32m8(__riscv_vle32_v_f32m8(y + i, vl), s, vl), vl);
        i += vl;
    }
}

// y = src * routing weight (or a plain copy without fused weights), one pass
static inline void mmid_store_row(float * y, const float * src, int64_t n, const ggml_tensor * w, int64_t i1, int64_t i2) {
    if (w == nullptr) {
        memcpy(y, src, n * sizeof(float));
        return;
    }
    if (!moe_store1_enabled()) {
        memcpy(y, src, n * sizeof(float));
        mmid_scale_row(y, n, w, i1, i2);
        return;
    }
    const float s = *(const float *) ((const char *) w->data + i1 * w->nb[1] + i2 * w->nb[2]);
    for (int64_t i = 0; i < n;) {
        const size_t vl = __riscv_vsetvl_e32m8(n - i);
        __riscv_vse32_v_f32m8(y + i, __riscv_vfmul_vf_f32m8(__riscv_vle32_v_f32m8(src + i, vl), s, vl), vl);
        i += vl;
    }
}

// MUL side: true if src0 already carries the weights (dst then only needs src0's values).
static bool mul_is_fused_moe_w(const ggml_tensor * op) {
    const ggml_tensor * m = op->src[0];
    return m->op == GGML_OP_MUL_MAT_ID && ggml_get_op_params_i32(m, FUSE_P_STATE) == 1 &&
           ggml_get_op_params_i32(m, FUSE_P_APPLIED) == 1 && fuse_get_ptr(m) == op->src[1];
}

// Debug: GGML_SPACEMIT_RVV_OFF=ADD,MUL,... (ggml_op_name) or ALL routes those ops to the generic CPU path.
static bool rvv_op_disabled(const ggml_tensor * op) {
    static const std::vector<bool> off = [] {
        std::vector<bool> v(GGML_OP_COUNT, false);
        const char * e = getenv("GGML_SPACEMIT_RVV_OFF");
        if (!e) {
            return v;
        }
        const std::string s = std::string(",") + e + ",";
        for (int i = 0; i < GGML_OP_COUNT; i++) {
            const std::string name = ggml_op_name((ggml_op) i);
            v[i] = s.find(",ALL,") != std::string::npos || s.find("," + name + ",") != std::string::npos;
        }
        return v;
    }();
    return off[op->op];
}

// extra MUL_MAT_ID workspace for the 4-row path: grouped q8 rows, per-expert offsets, group list, per-thread scratch
static size_t mmid_m4_work_size(int n_threads, const ggml_tensor * op, size_t nbw1) {
    const int64_t n_rows = op->src[2]->ne[0] * op->src[2]->ne[1];  // n_expert_used * n_tokens
    const int64_t n_as   = op->src[0]->ne[2];
    size_t        size   = 64;
    // per-expert alignment and up to 8 padding rows per expert (zero-padded m8 groups)
    size += GGML_PAD(n_rows * nbw1, 64) + n_as * (spacemit_kernels::q8_hp_al_align + 8 * nbw1);
    size += GGML_PAD(n_as * sizeof(int64_t), 64);
    size += GGML_PAD((n_rows / 4 + 1 + n_as) * 3 * sizeof(int32_t) + sizeof(int64_t), 64);
    size += (size_t) n_threads * GGML_PAD((8 * op->src[1]->ne[0] + 8 * op->src[0]->ne[1]) * sizeof(float), 64);
    return size;
}

static constexpr int     mmid_stats_max_threads = 64;
static mmid_phase_stats  g_mmid_stats[2][mmid_stats_max_threads];
static mmid_route_stats  g_mmid_route[2];
static std::once_flag    g_mmid_stats_once;
static dense_phase_stats g_dense_stats[3][mmid_stats_max_threads];
static std::once_flag    g_dense_stats_once;

static void dense_stats_dump() {
    const char * kinds[3] = { "q4_0", "q8_0", "other" };
    for (int k = 0; k < 3; k++) {
        dense_phase_stats sum = {};
        int               nth = 0;
        for (int i = 0; i < mmid_stats_max_threads; i++) {
            const dense_phase_stats & t = g_dense_stats[k][i];
            if (t.calls == 0) {
                continue;
            }
            nth++;
            sum.calls += t.calls;
            sum.t_total += t.t_total;
            sum.t_quant += t.t_quant;
            sum.t_sync += t.t_sync;
            sum.t_acopy += t.t_acopy;
            sum.t_bcopy += t.t_bcopy;
            sum.t_barrier += t.t_barrier;
            sum.t_gemm += t.t_gemm;
            for (int b = 0; b < 3; b++) {
                sum.branch[b] += t.branch[b];
            }
            sum.flops += t.flops;
            sum.b_bytes += t.b_bytes;
        }
        if (nth == 0 || sum.t_total == 0) {
            continue;
        }
        auto phase = [&](const char * name, int64_t ns) {
            fprintf(stderr, "  %-34s %10.1f ms/thread %6.2f%%\n", name, ns / 1e6 / nth, 100.0 * ns / sum.t_total);
        };
        fprintf(stderr, "\n== spacemit dense MUL_MAT sub-phases (%s, multi-row): calls=%lld threads=%d ==\n", kinds[k],
                (long long) (sum.calls / nth), nth);
        fprintf(stderr, "  %-34s %10.1f ms/thread\n", "total", sum.t_total / 1e6 / nth);
        phase("quantize A", sum.t_quant);
        phase("grid sync after quantize", sum.t_sync);
        phase("A copy to TCM", sum.t_acopy);
        phase("B copy to TCM", sum.t_bcopy);
        phase("pair barrier", sum.t_barrier);
        phase("gemm kernels", sum.t_gemm);
        phase("other", sum.t_total - sum.t_quant - sum.t_sync - sum.t_acopy - sum.t_bcopy - sum.t_barrier - sum.t_gemm);
        const double gemm_s = sum.t_gemm / 1e9 / nth;
        const double tot_s  = sum.t_total / 1e9 / nth;
        fprintf(stderr, "  gemm throughput: %.1f GOP/s over %d threads (whole op %.1f GOP/s)\n",
                gemm_s > 0 ? sum.flops / 1e9 / gemm_s : 0.0, nth, tot_s > 0 ? sum.flops / 1e9 / tot_s : 0.0);
        fprintf(stderr, "  weight bytes read by kernels: %.1f GB/s during gemm (%.2f GB total)\n",
                gemm_s > 0 ? sum.b_bytes / 1e9 / gemm_s : 0.0, sum.b_bytes / 1e9);
        fprintf(stderr, "  branch calls (per thread): A-in-TCM %lld, B-in-TCM %lld, generic %lld\n",
                (long long) (sum.branch[0] / nth), (long long) (sum.branch[1] / nth), (long long) (sum.branch[2] / nth));
    }
}

static void mmid_stats_dump() {
    const char * kinds[2] = { "1 token", "multi-token" };
    for (int k = 0; k < 2; k++) {
        const mmid_route_stats & r = g_mmid_route[k];
        if (r.calls == 0) {
            continue;
        }
        mmid_phase_stats sum = {};
        int              nth = 0;
        for (int i = 0; i < mmid_stats_max_threads; i++) {
            const mmid_phase_stats & t = g_mmid_stats[k][i];
            if (t.calls == 0) {
                continue;
            }
            nth++;
            sum.t_total += t.t_total;
            sum.t_quant += t.t_quant;
            sum.t_route += t.t_route;
            sum.t_sync += t.t_sync;
            sum.t_wcopy += t.t_wcopy;
            sum.t_agather += t.t_agather;
            sum.t_kernel += t.t_kernel;
            sum.rows_m4 += t.rows_m4;
            sum.t_quant4 += t.t_quant4;
            sum.rows_m2 += t.rows_m2;
            sum.rows_m1 += t.rows_m1;
            sum.w_tcm_hit += t.w_tcm_hit;
            sum.w_tcm_miss += t.w_tcm_miss;
            sum.fast_path += t.fast_path;
            sum.kernel_flops += t.kernel_flops;
            sum.t_k8 += t.t_k8;
            sum.t_k4 += t.t_k4;
            sum.t_scatter += t.t_scatter;
            sum.t_q4wait += t.t_q4wait;
            sum.t_scatter_wide += t.t_scatter_wide;
            sum.rows_m8 += t.rows_m8;
            sum.flops_m8 += t.flops_m8;
            sum.flops_m4 += t.flops_m4;
        }
        if (nth == 0) {
            continue;
        }
        const double tot   = sum.t_total / 1e6 / nth;
        auto         phase = [&](const char * name, int64_t ns) {
            fprintf(stderr, "  %-34s %10.1f ms/thread %6.2f%%\n", name, ns / 1e6 / nth, 100.0 * ns / sum.t_total);
        };
        fprintf(stderr, "\n== spacemit MUL_MAT_ID sub-phases (%s graphs): calls=%lld threads=%d tcm_size=%lld B ==\n",
                kinds[k], (long long) r.calls, nth, (long long) r.tcm_size);
        fprintf(stderr, "  %-34s %10.1f ms/thread\n", "total", tot);
        phase("quantize activations", sum.t_quant);
        phase("routing (thread 0 only)", sum.t_route);
        phase("grid sync after routing", sum.t_sync);
        phase("8/4-row quantize", sum.t_quant4);
        phase("grid sync after 8/4-row quantize", sum.t_q4wait);
        phase("weight copy to TCM", sum.t_wcopy);
        phase("activation gather to TCM", sum.t_agather);
        phase("gemm m8", sum.t_k8);
        phase("gemm m4", sum.t_k4);
        phase("m8/m4 output scatter + scale", sum.t_scatter);
        phase("  of which ne01 > 1024 (down)", sum.t_scatter_wide);
        phase("gemm m2/m1 + bookkeeping", sum.t_kernel);
        phase("other", sum.t_total - sum.t_quant - sum.t_route - sum.t_sync - sum.t_quant4 - sum.t_q4wait -
                           sum.t_wcopy - sum.t_agather - sum.t_kernel - sum.t_k8 - sum.t_k4 - sum.t_scatter);
        auto tput = [&](const char * name, double flops, int64_t ns) {
            const double s = ns / 1e9 / nth;
            fprintf(stderr, "  %-34s %10.1f GOP/s over %d threads\n", name, s > 0 ? flops / 1e9 / s : 0.0, nth);
        };
        tput("m8 kernel throughput", sum.flops_m8, sum.t_k8);
        tput("m4 kernel throughput", sum.flops_m4, sum.t_k4);
        tput("m2/m1 kernel throughput", sum.kernel_flops - sum.flops_m8 - sum.flops_m4, sum.t_kernel);
        fprintf(stderr, "  rows via m8 kernel: %lld\n", (long long) sum.rows_m8);
        const double kernel_s = (sum.t_kernel + sum.t_k8 + sum.t_k4 + sum.t_scatter) / 1e9 / nth;
        fprintf(stderr, "  kernel throughput: %.1f GOP/s over %d threads (%.2f GOP/s/thread)\n",
                kernel_s > 0 ? sum.kernel_flops / 1e9 / kernel_s : 0.0, nth,
                kernel_s > 0 ? sum.kernel_flops / 1e9 / kernel_s / nth : 0.0);
        fprintf(stderr, "  rows via m4 kernel (incl. m8): %lld, m2 kernel: %lld, m1 kernel: %lld (summed over threads)\n",
                (long long) sum.rows_m4, (long long) sum.rows_m2, (long long) sum.rows_m1);
        fprintf(stderr, "  expert weight slice in TCM: %lld, not in TCM: %lld, fast-path calls: %lld\n",
                (long long) sum.w_tcm_hit, (long long) sum.w_tcm_miss, (long long) sum.fast_path);
        fprintf(stderr, "  active experts/call: %.1f, rows/active expert: %.2f\n",
                (double) r.active_experts / r.calls, r.active_experts ? (double) r.rows / r.active_experts : 0.0);
        const char * labels[9] = { "1", "2-3", "4-7", "8-15", "16-31", "32-63", "64-127", "128-255", ">=256" };
        fprintf(stderr, "  rows/expert histogram:");
        for (int b = 0; b < 9; b++) {
            fprintf(stderr, " %s:%.1f%%", labels[b], r.active_experts ? 100.0 * r.hist[b] / r.active_experts : 0.0);
        }
        fprintf(stderr, "\n");
    }
}

template <typename BLOC_TYPE, int64_t INTER_SIZE, int64_t NB_COLS> class tensor_traits : public ggml::spacemit::tensor_traits_base {
    // Q4_0 hp: aligned 4-row A layout + signed int4 B (GGML_SPACEMIT_HP_ALIGNED, default on). The weights are
    // repacked accordingly at load time, so this must be decided once per process.
    static bool hp_aligned() {
        if constexpr (std::is_same_v<BLOC_TYPE, block_q4_0> && INTER_SIZE == 256) {
            return spacemit_kernels::hp_aligned_enabled();
        } else {
            return false;
        }
    }

    // Q8_0 pre-arranged path (GGML_SPACEMIT_Q8_PRE, default on): B in the vmadot i8 layout, 4-row A groups in the
    // aligned 192B layout (48B per row and K32 block; single rows keep 38B blocks, contiguous, in the same slots)
    static bool q8_pre() {
        if constexpr (std::is_same_v<BLOC_TYPE, block_q8_0>) {
            return spacemit_kernels::q8_prearranged_enabled();
        } else {
            return false;
        }
    }

    // the A workspace base is aligned to q8_hp_al_align
    static bool a_aligned() { return hp_aligned() || q8_pre(); }

    bool work_size(int n_threads, const ggml_tensor * op, size_t & size) const override {
        switch (op->op) {
            case GGML_OP_MUL_MAT:
                {
                    int64_t src1_nelements = ggml_nelements(op->src[1]);

                    if constexpr (std::is_same_v<BLOC_TYPE, block_q2_K> || std::is_same_v<BLOC_TYPE, block_q3_K>) {
                        size =
                            spacemit_kernels::div_round_up(src1_nelements, QK_K) * spacemit_kernels::q8k_blk_size(QK_K);
                    } else if constexpr (INTER_SIZE == QK4_0) {
                        size = spacemit_kernels::div_round_up(src1_nelements, QK4_0) *
                               (q8_pre() ? spacemit_kernels::q8_m4al_row_blk_size :
                                           spacemit_kernels::q8_blk_size(QK4_0, true));
                    } else if constexpr (INTER_SIZE == 256) {
                        size = spacemit_kernels::div_round_up(src1_nelements, 256) *
                               (hp_aligned() ? spacemit_kernels::q8_hp_al_row_blk_size :
                                               spacemit_kernels::q8_hp_blk_size(256, true, true));
                    } else {
                        GGML_ABORT("unsupported block type");
                    }

                    size = GGML_PAD(size, sizeof(int64_t));
                    if (a_aligned()) {
                        // room to align the workspace base and pad the A buffer (B may follow it in TCM)
                        size = GGML_PAD(size, spacemit_kernels::q8_hp_al_align) + spacemit_kernels::q8_hp_al_align;
                    }

                    return true;
                }
            case GGML_OP_MUL_MAT_ID:
                {
                    int64_t src1_nelements = ggml_nelements(op->src[1]);

                    if constexpr (std::is_same_v<BLOC_TYPE, block_q2_K> || std::is_same_v<BLOC_TYPE, block_q3_K>) {
                        size =
                            spacemit_kernels::div_round_up(src1_nelements, QK_K) * spacemit_kernels::q8k_blk_size(QK_K);
                    } else if constexpr (INTER_SIZE == QK4_0) {
                        size = spacemit_kernels::div_round_up(src1_nelements, QK4_0) *
                               spacemit_kernels::q8_blk_size(QK4_0, true);
                    } else if constexpr (INTER_SIZE == 256) {
                        size = spacemit_kernels::div_round_up(src1_nelements, 256) *
                               spacemit_kernels::q8_hp_blk_size(256, true, true);
                    } else {
                        GGML_ABORT("unsupported block type");
                    }

                    size = GGML_PAD(size, sizeof(int64_t));

                    const int64_t ne02 = op->src[0]->ne[2];  // n_as, n_expert
                    const int64_t ne12 = op->src[1]->ne[2];  // n_tokens

                    const size_t sizeof_mmid_row_mapping = sizeof(int64_t);
                    size += sizeof_mmid_row_mapping * ne02 * (ne12 + 1) + (ne02 + 1) * sizeof(int64_t);

                    size = GGML_PAD(size, sizeof(int64_t));

                    if constexpr (INTER_SIZE == 256 && !std::is_same_v<BLOC_TYPE, block_q2_K> &&
                                  !std::is_same_v<BLOC_TYPE, block_q3_K>) {
                        const size_t nbw1 = spacemit_kernels::div_round_up(op->src[1]->ne[0], 256) *
                                            (hp_aligned() ? spacemit_kernels::q8_hp_al_row_blk_size :
                                                            spacemit_kernels::q8_hp_blk_size(256, true, true));
                        size += mmid_m4_work_size(n_threads, op, nbw1);
                        if (hp_aligned()) {
                            size += spacemit_kernels::q8_hp_al_align;  // 128B alignment of the 4-row groups
                        }
                    }

                    return true;
                }
            default:
                // GGML_ABORT("fatal error");
                break;
        }
        return false;
    }

    bool compute_forward(ggml::spacemit::context & ctx, ggml_tensor * op) const override {
        switch (op->op) {
            case GGML_OP_MUL_MAT:
                switch (op->src[0]->type) {
                    case GGML_TYPE_Q2_K:
                    case GGML_TYPE_Q3_K:
                    case GGML_TYPE_Q4_0:
                    case GGML_TYPE_Q4_1:
                    case GGML_TYPE_Q4_K:
                    case GGML_TYPE_Q6_K:
                    case GGML_TYPE_Q8_0:
                    case GGML_TYPE_Q5_0:
                    case GGML_TYPE_Q5_1:
                    case GGML_TYPE_Q5_K:
                        //case GGML_TYPE_MXFP4:
                        forward_mul_mat(ctx, op);
                        return true;
                    case GGML_TYPE_F32:
                        // F32 (router) weights converted to Q8_0 at load (tensor_traits_f32_as_q8_0)
                        if constexpr (std::is_same_v<BLOC_TYPE, block_q8_0>) {
                            forward_mul_mat(ctx, op);
                            return true;
                        }
                        return false;
                    default:
                        // GGML_ABORT("fatal error: unsupported type for src0 in MUL_MAT");
                        return false;
                }
                break;
            case GGML_OP_MUL_MAT_ID:
                switch (op->src[0]->type) {
                    case GGML_TYPE_Q2_K:
                    case GGML_TYPE_Q3_K:
                    case GGML_TYPE_Q4_0:
                    case GGML_TYPE_Q4_1:
                    case GGML_TYPE_Q4_K:
                    case GGML_TYPE_Q6_K:
                    case GGML_TYPE_Q8_0:
                    case GGML_TYPE_Q5_0:
                    case GGML_TYPE_Q5_1:
                    case GGML_TYPE_Q5_K:
                        //case GGML_TYPE_MXFP4:
                        forward_mul_mat_id(ctx, op);
                        return true;
                    default:
                        // GGML_ABORT("fatal error: unsupported type for src0 in MUL_MAT_ID");
                        return false;
                }
                break;
            default:
                // GGML_ABORT("fatal error");
                break;
        }
        return false;
    }

    void forward_mul_mat(ggml::spacemit::context & ctx, ggml_tensor * op) const {
        constexpr size_t a_blk_len = INTER_SIZE;
        constexpr size_t b_blk_len = INTER_SIZE;

        const ggml_tensor * src0 = op->src[0];
        const ggml_tensor * src1 = op->src[1];
        ggml_tensor *       dst  = op;

        GGML_TENSOR_BINARY_OP_LOCALS

        int ith = ctx.ith;
        int nth = ctx.nth;

        [[maybe_unused]] const enum ggml_type type = src0->type;

        void *        w_data  = (void *) src0->data;
        const float * feature = (const float *) src1->data;
        float *       output  = (float *) dst->data;

        const int64_t gemm_m = ne11 * ne12 * ne13;
        const int64_t gemm_k = ne10;
        const int64_t gemm_n = ne01;

        spacemit_kernels::quantize_a_row_def       quantize_a_row_i8;
        spacemit_kernels::quantize_a_row_def       quantize_a_4row_i8;
        spacemit_kernels::gemm_kernel_quantize_def gemm_kernel;
        bool                                       set_kernel_impl = false;
        bool                                       use_m8          = false;  // 8-row A groups (Q4_0 hp)

        int64_t block_stride_a = spacemit_kernels::q8_blk_size(a_blk_len);

#if defined(RISCV64_SPACEMIT_IME2)
        if (!set_kernel_impl && (global_spine_env_info.use_ime2)) {
            quantize_a_row_i8  = spacemit_kernels::rvv::quantize_a_row_i8;
            quantize_a_4row_i8 = spacemit_kernels::rvv::quantize_a_4row_i8;
            block_stride_a     = spacemit_kernels::q8_blk_size(a_blk_len, true);

            if constexpr (std::is_same_v<BLOC_TYPE, block_q6_K> || std::is_same_v<BLOC_TYPE, block_q8_0>) {
                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8i8;
                set_kernel_impl = true;
                if constexpr (std::is_same_v<BLOC_TYPE, block_q8_0>) {
                    if (spacemit_kernels::q8_prearranged_enabled()) {
                        // B pre-arranged at repack; 4-row A groups in 192B K32 blocks (data 64B aligned)
                        gemm_kernel        = spacemit_kernels::ime2::gemm_kernel_i8i8_pre_al;
                        quantize_a_4row_i8 = spacemit_kernels::rvv::quantize_a_4row_i8_al;
                        block_stride_a     = spacemit_kernels::q8_m4al_row_blk_size;
                    }
                }
            } else if constexpr (std::is_same_v<BLOC_TYPE, block_q4_0> || std::is_same_v<BLOC_TYPE, block_q4_1> ||
                                 std::is_same_v<BLOC_TYPE, block_q4_K>) {
                if constexpr (INTER_SIZE == 256) {
                    gemm_kernel        = spacemit_kernels::ime2::gemm_kernel_i8i4_hp;
                    quantize_a_row_i8  = spacemit_kernels::rvv::quantize_a_row_i8_hp;
                    quantize_a_4row_i8 = spacemit_kernels::rvv::quantize_a_4row_i8_hp;
                    block_stride_a     = spacemit_kernels::q8_hp_blk_size(a_blk_len, true, true);
                    if (hp_aligned()) {
                        // 4-row groups take 4 * 304B per K256; single rows keep the 290B layout in a 304B slot
                        gemm_kernel        = spacemit_kernels::ime2::gemm_kernel_i8i4_hp_al;
                        quantize_a_4row_i8 = spacemit_kernels::rvv::quantize_a_4row_i8_hp_al;
                        block_stride_a     = spacemit_kernels::q8_hp_al_row_blk_size;
                        if constexpr (std::is_same_v<BLOC_TYPE, block_q4_0>) {
                            // 8-row groups (2176B per K256) fit in 8 row slots (8 * 304B), 128B aligned
                            if (spacemit_kernels::hp_m8_enabled()) {
                                gemm_kernel = spacemit_kernels::ime2::gemm_kernel_i8i4_hp_m8_al;
                                use_m8      = true;
                            }
                        }
                    }
                    set_kernel_impl    = true;
                } else {
                    gemm_kernel        = spacemit_kernels::ime2::gemm_kernel_i8i4;
                    quantize_a_row_i8  = spacemit_kernels::rvv::quantize_a_row_i8;
                    quantize_a_4row_i8 = spacemit_kernels::rvv::quantize_a_4row_i8;
                    block_stride_a     = spacemit_kernels::q8_blk_size(a_blk_len, true);
                    set_kernel_impl    = true;
                }
            } else if constexpr (std::is_same_v<BLOC_TYPE, block_q2_K>) {
                quantize_a_row_i8  = spacemit_kernels::rvv::quantize_a_row_i8k;
                quantize_a_4row_i8 = spacemit_kernels::rvv::quantize_a_4row_i8k;
                block_stride_a     = spacemit_kernels::q8k_blk_size(a_blk_len);

                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8i2k;
                set_kernel_impl = true;
            } else if constexpr (std::is_same_v<BLOC_TYPE, block_q3_K>) {
                quantize_a_row_i8  = spacemit_kernels::rvv::quantize_a_row_i8k;
                quantize_a_4row_i8 = spacemit_kernels::rvv::quantize_a_4row_i8k;
                block_stride_a     = spacemit_kernels::q8k_blk_size(a_blk_len);

                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8i3k;
                set_kernel_impl = true;
            } else if constexpr (std::is_same_v<BLOC_TYPE, block_mxfp4>) {
                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8mxfp4;
                set_kernel_impl = true;
            } else if constexpr (std::is_same_v<BLOC_TYPE, block_q5_1> || std::is_same_v<BLOC_TYPE, block_q5_K> ||
                                 std::is_same_v<BLOC_TYPE, block_q5_0>) {
                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8i5;
                set_kernel_impl = true;
            }
        }
#endif

#if defined(RISCV64_SPACEMIT_IME1)
        if (!set_kernel_impl && (global_spine_env_info.use_ime1)) {
            quantize_a_row_i8  = spacemit_kernels::ime1::quantize_a_row_i8;
            quantize_a_4row_i8 = spacemit_kernels::ime1::quantize_a_4row_i8;

            if constexpr (std::is_same_v<BLOC_TYPE, block_q4_0> || std::is_same_v<BLOC_TYPE, block_q4_1> ||
                          std::is_same_v<BLOC_TYPE, block_q4_K>) {
                gemm_kernel     = spacemit_kernels::ime1::gemm_kernel_i8i4;
                set_kernel_impl = true;
            }
        }
#endif
        if (!set_kernel_impl) {
            GGML_ABORT("no kernel implementation found for the block type");
        }

        const int64_t a_k_blks = spacemit_kernels::div_round_up(gemm_k, a_blk_len);
        const int64_t b_k_blks = spacemit_kernels::div_round_up(gemm_k, b_blk_len);

        const int64_t row_stride_a        = a_k_blks * block_stride_a;
        const int64_t ws_align            = a_aligned() ? spacemit_kernels::q8_hp_al_align : alignof(int64_t);
        const int64_t gemm_workspace_size = GGML_PAD(gemm_m * row_stride_a, ws_align);

        if (ith == 0 && ctx.workspace_size < gemm_workspace_size + (a_aligned() ? ws_align : 0)) {
            GGML_ABORT("wsize less than gemm_workspace_size");
        }

        uintptr_t ws_ptr = reinterpret_cast<uintptr_t>(ctx.workspace);
        if (a_aligned()) {
            ws_ptr = (ws_ptr + ws_align - 1) & ~(uintptr_t) (ws_align - 1);
        }

        const int64_t tcm_buffer_size = (int64_t) ctx.shared.size;
        void *        tcm_buffer      = ctx.shared.data;

        auto * quant_a_buffer = reinterpret_cast<uint8_t *>(ws_ptr);

        const int64_t row_align = use_m8 ? 8 : 4;
        // GGML_SPACEMIT_DENSE_MBLK=R (default 8, 1 = old behaviour): each thread runs R consecutive row groups
        // per pass over B, so a B tile is read once per R groups instead of once per group
        static const int64_t dense_mblk = [] {
            const char * e = getenv("GGML_SPACEMIT_DENSE_MBLK");
            return e ? std::max<int64_t>(1, atoll(e)) : (int64_t) 8;
        }();
        const int64_t     row_blks  = spacemit_kernels::div_round_up(gemm_m, row_align);

        const int64_t row_stride_b      = b_k_blks * get_repacked_block_type_size<BLOC_TYPE, INTER_SIZE, NB_COLS>();
        const int64_t per_mb_rows_wsize = row_align * row_stride_a;
        const int64_t per_nb_cols_wsize = NB_COLS * row_stride_b;

        const int64_t barrier_idx = static_cast<int64_t>(ith / 2);

        const bool          dstats = gemm_m > 1 && ggml_op_stats_enabled() && ith < mmid_stats_max_threads;
        dense_phase_stats * ds     = dstats ? &g_dense_stats[std::is_same_v<BLOC_TYPE, block_q4_0>   ? 0 :
                                                             std::is_same_v<BLOC_TYPE, block_q8_0> ? 1 :
                                                                                                     2][ith] :
                                              nullptr;
        const int64_t       ds_t0  = dstats ? ggml_op_stats_now_ns() : 0;
        int64_t             ds_t   = ds_t0;
        int64_t             ds_dummy = 0;
        auto                ds_lap = [&](int64_t & acc) {
            if (dstats) {
                const int64_t now = ggml_op_stats_now_ns();
                acc += now - ds_t;
                ds_t = now;
            }
        };
        struct ds_finish_t {
            std::function<void()> f;
            ~ds_finish_t() { f(); }
        } ds_finish{ [&] {
            if (dstats) {
                ds_lap(ds->t_gemm);
                ds->t_total += ds_t - ds_t0;
                ds->calls++;
            }
        } };
        if (dstats) {
            std::call_once(g_dense_stats_once, [] { atexit(dense_stats_dump); });
        }

        GGML_ASSERT(global_spine_env_info.init_barrier != nullptr);
        GGML_ASSERT(barrier_idx < static_cast<int64_t>(spine_init_barrier_count));
        spine_barrier_t * cur_barrier = &global_spine_env_info.init_barrier[barrier_idx];

        if (gemm_m == 1) {
            // the 1-row kernels walk K blocks contiguously, so split at the 1-row block size (block_stride_a is
            // the padded per-row slot size in the aligned hp and Q8_0 pre layouts)
            const int64_t blk_stride_1row =
                hp_aligned() ? (int64_t) spacemit_kernels::q8_hp_blk_size(a_blk_len, true, true) :
                q8_pre()     ? (int64_t) spacemit_kernels::q8_blk_size(a_blk_len, true) :
                               block_stride_a;
            int task_per_thread = spacemit_kernels::div_round_up(a_k_blks, nth);
            int a_blk_start     = ith * task_per_thread;
            int a_blk_end       = std::min(a_blk_start + task_per_thread, (int) a_k_blks);
            if (a_blk_start < a_blk_end) {
                quantize_a_row_i8(a_blk_len, feature + a_blk_start * a_blk_len, (a_blk_end - a_blk_start) * a_blk_len,
                                  quant_a_buffer + a_blk_start * blk_stride_1row);
            }
        } else {
            int task_per_thread = spacemit_kernels::div_round_up(row_blks, nth);
            int m_row_blk_start = ith * task_per_thread;
            int m_row_blk_end   = std::min(m_row_blk_start + task_per_thread, (int) row_blks);
            for (int m_row_blk = m_row_blk_start; m_row_blk < m_row_blk_end; m_row_blk++) {
                int m_idx             = m_row_blk * row_align;
                int rows_tobe_handled = (gemm_m - m_idx) > row_align ? row_align : (gemm_m - m_idx);

                if (use_m8 && rows_tobe_handled == 8) {
                    spacemit_kernels::rvv::quantize_a_8row_i8_hp_m8(a_blk_len, feature + m_idx * gemm_k, gemm_k,
                                                                    quant_a_buffer + m_idx * row_stride_a);
                } else if (rows_tobe_handled == row_align && quantize_a_4row_i8 != nullptr) {
                    const float * a_row_ptr       = feature + m_idx * gemm_k;
                    auto *        quant_a_row_ptr = quant_a_buffer + m_idx * row_stride_a;
                    quantize_a_4row_i8(a_blk_len, a_row_ptr, gemm_k, quant_a_row_ptr);
                } else {
                    // tail: one 4-row group if possible (gemm_kernel_i8i4_hp_m8_al falls back the same way)
                    if (use_m8 && rows_tobe_handled >= 4) {
                        quantize_a_4row_i8(a_blk_len, feature + m_idx * gemm_k, gemm_k,
                                           quant_a_buffer + m_idx * row_stride_a);
                        rows_tobe_handled -= 4;
                        m_idx += 4;
                    }
                    while (rows_tobe_handled) {
                        const float * a_row_ptr       = feature + m_idx * gemm_k;
                        auto *        quant_a_row_ptr = quant_a_buffer + m_idx * row_stride_a;
                        quantize_a_row_i8(a_blk_len, a_row_ptr, gemm_k, quant_a_row_ptr);
                        rows_tobe_handled -= 1;
                        m_idx += 1;
                    }
                }
            }
        }

        ds_lap(dstats ? ds->t_quant : ds_dummy);
        ctx.sync();
        ds_lap(dstats ? ds->t_sync : ds_dummy);

        const int64_t gemm_m_stride     = gemm_n / gemm_m > 64 ? gemm_m : 16;
        const int64_t gemm_m_blocked    = spacemit_kernels::div_round_up(gemm_m, gemm_m_stride);
        const int64_t max_gemm_n_stride = spacemit_kernels::div_round_up(gemm_n * gemm_m_blocked, nth);

        int64_t gemm_n_stride = gemm_n;
        if (max_gemm_n_stride < gemm_n) {
            gemm_n_stride =
                std::min(gemm_n_stride, spacemit_kernels::div_round_up(max_gemm_n_stride, NB_COLS) * NB_COLS);
        }

        if (gemm_n_stride == gemm_n && tcm_buffer != nullptr && per_mb_rows_wsize <= tcm_buffer_size) {
            if (dstats) {
                ds->branch[0]++;
            }
            int64_t m_blk = row_align;
            if (dense_mblk > 1 && gemm_m > row_align) {
                // largest R' <= R that still gives every thread a block and fits TCM
                for (int64_t r = dense_mblk; r >= 1; r--) {
                    if (r * row_align * row_stride_a <= tcm_buffer_size &&
                        spacemit_kernels::div_round_up(gemm_m, r * row_align) >= nth) {
                        m_blk = r * row_align;
                        break;
                    }
                }
            }
            for (int64_t m_start = ith * m_blk; m_start < gemm_m; m_start += m_blk * nth) {
                uint8_t * b_col    = reinterpret_cast<uint8_t *>(w_data);
                uint8_t * b_col_zp = block_type_has_zp<BLOC_TYPE>() ? b_col : nullptr;

                int64_t m_row_real = std::min(gemm_m - m_start, m_blk);

                spacemit_kernels::rvv::memcpy1d(tcm_buffer, quant_a_buffer + m_start * row_stride_a,
                                                m_row_real * row_stride_a);
                if (dstats) {
                    ds_lap(ds->t_acopy);
                    ds->flops += 2.0 * m_row_real * gemm_n * gemm_k;
                    ds->b_bytes += (double) gemm_n * row_stride_b;
                }

                int64_t n_blk_real = 0;
                for (int64_t ni = 0; ni < gemm_n; ni += n_blk_real, b_col += n_blk_real * row_stride_b) {
                    n_blk_real = std::min(gemm_n - ni, (int64_t) NB_COLS);

                    uint8_t * a_row_ptr = (uint8_t *) tcm_buffer;
                    float *   c_blk     = output + m_start * gemm_n + ni;

                    int32_t rows_remaining = m_row_real;

                    while (rows_remaining > 0) {
                        auto rows_handled = gemm_kernel(b_blk_len, a_row_ptr, b_col, b_col_zp, c_blk, rows_remaining,
                                                        n_blk_real, b_k_blks, gemm_n);

                        c_blk += rows_handled * gemm_n;
                        a_row_ptr += rows_handled * row_stride_a;

                        rows_remaining -= rows_handled;
                    }
                }
                ds_lap(dstats ? ds->t_gemm : ds_dummy);
            }
        } else if (tcm_buffer != nullptr && per_nb_cols_wsize <= tcm_buffer_size) {
            if (dstats) {
                ds->branch[1]++;
            }
            uint8_t * a_row = quant_a_buffer;
            uint8_t * b_col = reinterpret_cast<uint8_t *>(tcm_buffer);
            if ((gemm_workspace_size + per_nb_cols_wsize) <= tcm_buffer_size) {
                a_row = (uint8_t *) tcm_buffer;
                b_col = reinterpret_cast<uint8_t *>(tcm_buffer) + gemm_workspace_size;
            }
            uint8_t * b_col_zp = block_type_has_zp<BLOC_TYPE>() ? b_col : nullptr;

            // Direct IME-from-DRAM GEMV path, decode only (gemm_m == 1), q4_0 only:
            // q8_0 always loses to the staged TCM path here.
            if constexpr (std::is_same_v<BLOC_TYPE, block_q4_0> && INTER_SIZE == 256 && NB_COLS == 32) {
                if (gemm_m == 1 && a_row != quant_a_buffer) {
                    spacemit_kernels::rvv::memcpy1d(a_row, quant_a_buffer, gemm_workspace_size);
                    constexpr int64_t tile_cols = 4 * NB_COLS;
                    for (int64_t ni = (int64_t) ith * tile_cols; ni < gemm_n; ni += tile_cols * nth) {
                        const int64_t nb_real  = std::min(gemm_n - ni, tile_cols);
                        uint8_t *     b_row    = reinterpret_cast<uint8_t *>(w_data) + ni * row_stride_b;
                        uint8_t *     b_row_zp = block_type_has_zp<BLOC_TYPE>() ? b_row : nullptr;
                        gemm_kernel(b_blk_len, a_row, b_row, b_row_zp, output + ni, 1, nb_real, b_k_blks, gemm_n);
                    }
                    return;
                }
            }

            const bool has_pair = ((ith & 1) != 0) || (ith + 1 < nth);

            const int64_t ni0     = (int64_t) ith * NB_COLS;
            const bool    active0 = ni0 < gemm_n;
            const int64_t nb0     = active0 ? std::min(gemm_n - ni0, (int64_t) NB_COLS) : 0;

            if (ith % 2 == 0 && active0) {
                spacemit_kernels::rvv::memcpy1d(b_col, reinterpret_cast<uint8_t *>(w_data) + ni0 * row_stride_b,
                                                nb0 * row_stride_b);
                if (a_row != quant_a_buffer) {
                    spacemit_kernels::rvv::memcpy1d(a_row, quant_a_buffer, gemm_workspace_size);
                }
            }

            if (has_pair) {
                spine_barrier_wait(cur_barrier);
            }

            if (ith % 2 != 0 && active0) {
                if (a_row != quant_a_buffer) {
                    spacemit_kernels::rvv::memcpy1d(a_row, quant_a_buffer, gemm_workspace_size);
                }
                spacemit_kernels::rvv::memcpy1d(b_col, reinterpret_cast<uint8_t *>(w_data) + ni0 * row_stride_b,
                                                nb0 * row_stride_b);
            }

            const int64_t base_start = (int64_t) (ith & ~1) * NB_COLS;
            const int64_t lane_off   = (int64_t) (ith & 1) * NB_COLS;
            for (int64_t base = base_start; base < gemm_n; base += NB_COLS * nth) {
                const int64_t ni      = base + lane_off;
                const bool    active  = ni < gemm_n;
                const int64_t nb_real = active ? std::min(gemm_n - ni, (int64_t) NB_COLS) : 0;

                if (has_pair && ith % 2 != 0) {
                    spine_barrier_wait(cur_barrier);
                }

                if (active) {
                    int64_t rows_remaining = gemm_m;
                    float * c_blk          = output + ni;
                    auto *  a_row_cur      = a_row;

                    while (rows_remaining > 0) {
                        auto rows_handled = gemm_kernel(b_blk_len, a_row_cur, b_col, b_col_zp, c_blk, rows_remaining,
                                                        nb_real, b_k_blks, gemm_n);

                        c_blk += rows_handled * gemm_n;
                        a_row_cur += rows_handled * row_stride_a;

                        rows_remaining -= rows_handled;
                    }
                }

                if (has_pair && ith % 2 == 0) {
                    spine_barrier_wait(cur_barrier);
                }

                const int64_t next_ni = ni + NB_COLS * nth;
                if (next_ni < gemm_n) {
                    const int64_t next_nb = std::min(gemm_n - next_ni, (int64_t) NB_COLS);
                    spacemit_kernels::rvv::memcpy1d(b_col, reinterpret_cast<uint8_t *>(w_data) + next_ni * row_stride_b,
                                                    next_nb * row_stride_b);
                }
            }
        } else {
            if (dstats) {
                ds->branch[2]++;
            }
            const int64_t task_count_m = spacemit_kernels::div_round_up(gemm_m, gemm_m_stride);
            const int64_t task_count_n = spacemit_kernels::div_round_up(gemm_n, gemm_n_stride);

            int64_t task_count      = task_count_m * task_count_n;
            int64_t task_per_thread = (task_count + nth - 1) / nth;
            int64_t start           = ith * task_per_thread;
            int64_t end             = std::min((ith + 1) * task_per_thread, task_count);
            for (int64_t compute_idx = start; compute_idx < end; compute_idx++) {
                const auto tid_n = compute_idx / task_count_m;
                const auto tid_m = compute_idx % task_count_m;

                const int64_t m_start = tid_m * gemm_m_stride;
                const int64_t m_count = std::min(gemm_m - m_start, (int64_t) gemm_m_stride);

                const int64_t n_start = tid_n * gemm_n_stride;
                const int64_t n_count = std::min(gemm_n - n_start, (int64_t) gemm_n_stride);

                const int64_t n_blk = m_count == 1 ? n_count : NB_COLS;

                uint8_t * b_col    = reinterpret_cast<uint8_t *>(w_data) + n_start * row_stride_b;
                uint8_t * b_col_zp = block_type_has_zp<BLOC_TYPE>() ? b_col : nullptr;

                int64_t n_blk_real = 0;
                for (int64_t ni = 0; ni < n_count; ni += n_blk_real, b_col += n_blk_real * row_stride_b) {
                    n_blk_real = std::min(n_count - ni, n_blk);

                    uint8_t * a_row = quant_a_buffer + m_start * row_stride_a;

                    float * c_blk = output + m_start * gemm_n + n_start + ni;

                    int64_t rows_remaining = m_count;

                    uint8_t * b_col_cur    = b_col;
                    uint8_t * b_col_zp_cur = b_col_zp;

                    while (rows_remaining > 0) {
                        auto rows_handled = gemm_kernel(b_blk_len, a_row, b_col_cur, b_col_zp_cur, c_blk,
                                                        rows_remaining, n_blk_real, b_k_blks, gemm_n);

                        c_blk += rows_handled * gemm_n;
                        a_row += rows_handled * row_stride_a;

                        rows_remaining -= rows_handled;
                    }
                }
            }
        }
    }

    void forward_mul_mat_id(ggml::spacemit::context & ctx, ggml_tensor * op) const {
        constexpr size_t a_blk_len = INTER_SIZE;
        constexpr size_t b_blk_len = INTER_SIZE;

        const ggml_tensor * src0 = op->src[0];
        const ggml_tensor * src1 = op->src[1];
        const ggml_tensor * ids  = op->src[2];
        ggml_tensor *       dst  = op;

        GGML_TENSOR_BINARY_OP_LOCALS

        int ith = ctx.ith;
        int nth = ctx.nth;

        // row groups
        const int n_ids = ids->ne[0];  // n_expert_used
        const int n_as  = ne02;        // n_expert

        const ggml_tensor * fw = mmid_fused_weights(op);  // routing weights folded into the output, or nullptr

        const bool         op_stats = ggml_op_stats_enabled() && ith < mmid_stats_max_threads;
        const int          st_kind  = ids->ne[1] > 1 ? 1 : 0;
        mmid_phase_stats * st       = op_stats ? &g_mmid_stats[st_kind][ith] : nullptr;
        const int64_t      st_t0    = op_stats ? ggml_op_stats_now_ns() : 0;
        int64_t            st_t     = st_t0;
        auto               st_lap   = [&](int64_t & acc) {
            if (op_stats) {
                const int64_t now = ggml_op_stats_now_ns();
                acc += now - st_t;
                st_t = now;
            }
        };
        if (op_stats) {
            std::call_once(g_mmid_stats_once, [] { atexit(mmid_stats_dump); });
            st->calls++;
        }

        struct mmid_row_mapping {
            int32_t i1;
            int32_t i2;
        };

        spacemit_kernels::quantize_a_row_def           quantize_a_row_i8;
        spacemit_kernels::gemm_kernel_quantize_def     gemm_kernel;
        spacemit_kernels::moe_gemm_kernel_quantize_def moe_gemm_kernel_m2 = nullptr;
        spacemit_kernels::quantize_a_row_def           quantize_a_4row_i8 = nullptr;
        bool                                           set_kernel_impl = false;
        size_t                                         block_stride_a  = spacemit_kernels::q8_blk_size(QK4_0);

#if defined(RISCV64_SPACEMIT_IME2)
        if (!set_kernel_impl && (global_spine_env_info.use_ime2)) {
            quantize_a_row_i8 = spacemit_kernels::rvv::quantize_a_row_i8;
            block_stride_a    = spacemit_kernels::q8_blk_size(QK4_0, true);

            if constexpr (std::is_same_v<BLOC_TYPE, block_q6_K> || std::is_same_v<BLOC_TYPE, block_q8_0>) {
                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8i8;
                set_kernel_impl = true;
                if constexpr (std::is_same_v<BLOC_TYPE, block_q8_0>) {
                    if (spacemit_kernels::q8_prearranged_enabled()) {
                        gemm_kernel = spacemit_kernels::ime2::gemm_kernel_i8i8_pre;  // weights pre-arranged at repack
                    }
                }
            } else if constexpr (std::is_same_v<BLOC_TYPE, block_q4_0> || std::is_same_v<BLOC_TYPE, block_q4_1> ||
                                 std::is_same_v<BLOC_TYPE, block_q4_K>) {
                if constexpr (INTER_SIZE == 256) {
                    gemm_kernel       = spacemit_kernels::ime2::gemm_kernel_i8i4_hp;
                    quantize_a_row_i8 = spacemit_kernels::rvv::quantize_a_row_i8_hp;
                    block_stride_a    = spacemit_kernels::q8_hp_blk_size(a_blk_len, true, true);
                    if (mmid_m4_enabled()) {
                        quantize_a_4row_i8 = spacemit_kernels::rvv::quantize_a_4row_i8_hp;
                    }
                    if (hp_aligned()) {
                        gemm_kernel = spacemit_kernels::ime2::gemm_kernel_i8i4_hp_al;
                        if (mmid_m4_enabled()) {
                            quantize_a_4row_i8 = spacemit_kernels::rvv::quantize_a_4row_i8_hp_al;
                        }
                    }
                    set_kernel_impl = true;
                } else {
                    gemm_kernel        = spacemit_kernels::ime2::gemm_kernel_i8i4;
                    moe_gemm_kernel_m2 = spacemit_kernels::ime2::moe_m2_gemm_kernel_i8i4;
                    quantize_a_row_i8  = spacemit_kernels::rvv::quantize_a_row_i8;
                    block_stride_a     = spacemit_kernels::q8_blk_size(a_blk_len, true);
                    set_kernel_impl    = true;
                }
            } else if constexpr (std::is_same_v<BLOC_TYPE, block_q2_K>) {
                quantize_a_row_i8 = spacemit_kernels::rvv::quantize_a_row_i8k;
                block_stride_a    = spacemit_kernels::q8k_blk_size(a_blk_len);
                gemm_kernel       = spacemit_kernels::ime2::gemm_kernel_i8i2k;
                set_kernel_impl   = true;
            } else if constexpr (std::is_same_v<BLOC_TYPE, block_q3_K>) {
                quantize_a_row_i8 = spacemit_kernels::rvv::quantize_a_row_i8k;
                block_stride_a    = spacemit_kernels::q8k_blk_size(a_blk_len);
                gemm_kernel       = spacemit_kernels::ime2::gemm_kernel_i8i3k;
                set_kernel_impl   = true;
            } else if constexpr (std::is_same_v<BLOC_TYPE, block_mxfp4>) {
                gemm_kernel        = spacemit_kernels::ime2::gemm_kernel_i8mxfp4;
                moe_gemm_kernel_m2 = spacemit_kernels::ime2::moe_m2_gemm_kernel_i8mxfp4;
                set_kernel_impl    = true;
            } else if constexpr (std::is_same_v<BLOC_TYPE, block_q5_1> || std::is_same_v<BLOC_TYPE, block_q5_K> ||
                                 std::is_same_v<BLOC_TYPE, block_q5_0>) {
                gemm_kernel        = spacemit_kernels::ime2::gemm_kernel_i8i5;
                moe_gemm_kernel_m2 = spacemit_kernels::ime2::moe_m2_gemm_kernel_i8i5;
                set_kernel_impl    = true;
            }
        }
#endif

#if defined(RISCV64_SPACEMIT_IME1)
        if (!set_kernel_impl && (global_spine_env_info.use_ime1)) {
            quantize_a_row_i8 = spacemit_kernels::ime1::quantize_a_row_i8;

            if constexpr (std::is_same_v<BLOC_TYPE, block_q4_0> || std::is_same_v<BLOC_TYPE, block_q4_1> ||
                          std::is_same_v<BLOC_TYPE, block_q4_K>) {
                gemm_kernel     = spacemit_kernels::ime1::gemm_kernel_i8i4;
                set_kernel_impl = true;
            }
        }
#endif
        if (!set_kernel_impl) {
            GGML_ABORT("no kernel implementation found for the block type");
        }

        const size_t a_k_blks = spacemit_kernels::div_round_up(ne10, a_blk_len);
        const size_t b_k_blks = spacemit_kernels::div_round_up(ne10, b_blk_len);

        const size_t nbw1                = a_k_blks * block_stride_a;
        const size_t nbw2                = ne11 * nbw1;
        const size_t nbw3                = nbw2 * ne12;
        const size_t gemm_workspace_size = GGML_PAD(nbw3, alignof(int64_t));

        const uintptr_t ws_ptr         = reinterpret_cast<uintptr_t>(ctx.workspace);
        auto *          quant_a_buffer = reinterpret_cast<uint8_t *>(ws_ptr);

        // every row goes through the 8-row groups (zero-padded tail): the 1-row quantized copy is not needed
        bool pad8_all = false;
        if constexpr (std::is_same_v<BLOC_TYPE, block_q4_0> && INTER_SIZE == 256) {
            pad8_all = ids->ne[1] > 1 && quantize_a_4row_i8 != nullptr && ne01 % 32 == 0 && hp_aligned() &&
                       spacemit_kernels::hp_m8_enabled() && moe_pad8_enabled();
        }

        // routing and 8-row groups already in the work buffer (previous node, same inputs); needs pad8_all on both
        const bool reuse_q = pad8_all && ggml_get_op_params_i32(op, FUSE_P_REUSEQ) == 1;

        if (pad8_all) {
            // nothing: rows are quantized per 8-row group below
        } else if (ne11 == 1) {
            for (int64_t ii = ith; ii < ne12 * a_k_blks; ii += nth) {
                int64_t i12       = ii / a_k_blks;
                int64_t ak_blk_id = ii % a_k_blks;
                quantize_a_row_i8(a_blk_len, (float *) ((char *) src1->data + i12 * nb12) + ak_blk_id * a_blk_len,
                                  a_blk_len, quant_a_buffer + i12 * nbw2 + ak_blk_id * block_stride_a);
            }
        } else {
            for (int64_t ii = ith; ii < ne12 * ne11; ii += nth) {
                int64_t i12 = ii / ne11;
                int64_t i11 = ii % ne11;
                quantize_a_row_i8(a_blk_len, (float *) ((char *) src1->data + i12 * nb12 + i11 * nb11), ne10,
                                  quant_a_buffer + i12 * nbw2 + i11 * nbw1);
            }
        }

#define MMID_MATRIX_ROW(row_id, i1) matrix_rows[(row_id) *ne12 + (i1)]

        st_lap(op_stats ? st->t_quant : st_t);

        int64_t *          matrix_row_counts       = (int64_t *) (ws_ptr + gemm_workspace_size);
        int32_t *          valid_ep_count          = (int32_t *) (matrix_row_counts + n_as);
        int32_t *          valid_act_count         = (int32_t *) (valid_ep_count + 1);
        int64_t *          valid_matrix_row_counts = (int64_t *) (valid_act_count + 1);
        mmid_row_mapping * matrix_rows             = (mmid_row_mapping *) (valid_matrix_row_counts + n_as);

        // 4-row path workspace (sized by mmid_m4_work_size)
        const bool use_m4  = quantize_a_4row_i8 != nullptr && ne01 % 32 == 0;
        // 8-row groups first, then at most one 4-row group per expert (signed-int4 Q4_0 only)
        bool use_m8 = false;
        if constexpr (std::is_same_v<BLOC_TYPE, block_q4_0> && INTER_SIZE == 256) {
            use_m8 = use_m4 && hp_aligned() && spacemit_kernels::hp_m8_enabled();
        }
        const size_t m8_grp_bytes = a_k_blks * spacemit_kernels::q8_hp_m8_blk_size;
        const int64_t n_rows_all = (int64_t) n_ids * ids->ne[1];
        auto align64 = [](uintptr_t p) { return (p + 63) & ~(uintptr_t) 63; };
        // per-row stride of the 4-row groups: 4 * nbw1_m4 bytes per group
        const size_t nbw1_m4 = hp_aligned() ? a_k_blks * spacemit_kernels::q8_hp_al_row_blk_size : nbw1;
        uintptr_t m4_ptr     = align64(reinterpret_cast<uintptr_t>(matrix_rows + n_as * ne12));
        if (hp_aligned()) {
            m4_ptr = (m4_ptr + spacemit_kernels::q8_hp_al_align - 1) & ~(uintptr_t) (spacemit_kernels::q8_hp_al_align - 1);
        }
        uint8_t * m4_quant         = reinterpret_cast<uint8_t *>(m4_ptr);
        m4_ptr = align64(m4_ptr + n_rows_all * nbw1_m4 + n_as * (spacemit_kernels::q8_hp_al_align + 8 * nbw1_m4));
        int64_t * m4_expert_offset = reinterpret_cast<int64_t *>(m4_ptr);  // byte offset of each expert's groups
        m4_ptr                     = align64(m4_ptr + n_as * sizeof(int64_t));
        int64_t * m4_n_groups      = reinterpret_cast<int64_t *>(m4_ptr);
        int32_t * m4_groups        = reinterpret_cast<int32_t *>(m4_n_groups + 1);  // (expert, first row, rows)
        m4_ptr = align64(m4_ptr + sizeof(int64_t) + (n_rows_all / 4 + 1 + n_as) * 3 * sizeof(int32_t));
        const size_t m4_scratch_sz = GGML_PAD((8 * ne10 + 8 * ne01) * sizeof(float), 64);
        float *      m4_a_f32      = reinterpret_cast<float *>(m4_ptr + ith * m4_scratch_sz);
        float *      m4_c_tmp      = m4_a_f32 + 8 * ne10;
        if (use_m4 && ith == 0) {
            GGML_ASSERT(m4_ptr + nth * m4_scratch_sz <= reinterpret_cast<uintptr_t>(ctx.workspace) + ctx.workspace_size);
        }

        if (ith == 0) {
            static FILE * ids_dump = [] {
                const char * e = getenv("GGML_SPACEMIT_DUMP_IDS");
                return e && e[0] ? fopen(e, "wb") : (FILE *) nullptr;
            }();
            if (ids_dump && ids->ne[1] > 1 && strstr(op->name, "ffn_moe_up") != nullptr && ids->type == GGML_TYPE_I32) {
                // record: name[64], ne0, ne1, then ne0 x ne1 int32 (rows contiguous in ne0)
                char name[64] = { 0 };
                snprintf(name, sizeof(name), "%s", op->name);
                const int32_t dims[2] = { (int32_t) ids->ne[0], (int32_t) ids->ne[1] };
                fwrite(name, 1, sizeof(name), ids_dump);
                fwrite(dims, sizeof(int32_t), 2, ids_dump);
                for (int64_t i1 = 0; i1 < ids->ne[1]; i1++) {
                    fwrite((const char *) ids->data + i1 * ids->nb[1], sizeof(int32_t), ids->ne[0], ids_dump);
                }
                fflush(ids_dump);
            }
        }

        if (ith == 0 && !reuse_q) {
            // initialize matrix_row_counts
            memset(matrix_row_counts, 0, n_as * sizeof(int64_t));

            // group rows by src0 matrix
            for (int32_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
                for (int32_t id = 0; id < n_ids; ++id) {
                    const int32_t i02 =
                        *(const int32_t *) ((const char *) ids->data + iid1 * ids->nb[1] + id * ids->nb[0]);

                    GGML_ASSERT(i02 >= 0 && i02 < n_as);

                    MMID_MATRIX_ROW(i02, matrix_row_counts[i02]) = { id, iid1 };
                    matrix_row_counts[i02] += 1;
                }
            }

            int32_t valid_ep_count_t  = 0;
            int32_t valid_act_count_t = 0;
            for (int cur_a = 0; cur_a < n_as; ++cur_a) {
                const int64_t cne1 = matrix_row_counts[cur_a];
                if (cne1 == 0) {
                    continue;
                }
                valid_matrix_row_counts[valid_ep_count_t] = cur_a;
                valid_act_count_t += cne1;
                valid_ep_count_t += 1;
            }
            valid_ep_count[0]  = valid_ep_count_t;
            valid_act_count[0] = valid_act_count_t;

            if (use_m4) {
                int64_t n_groups = 0;
                int64_t offset   = 0;
                for (int cur_a = 0; cur_a < n_as; ++cur_a) {
                    const int64_t cne1 = matrix_row_counts[cur_a];
                    offset = (offset + spacemit_kernels::q8_hp_al_align - 1) & ~(int64_t) (spacemit_kernels::q8_hp_al_align - 1);
                    m4_expert_offset[cur_a] = offset;
                    int64_t r = 0;
                    for (; use_m8 && r + 8 <= cne1; r += 8) {
                        m4_groups[3 * n_groups + 0] = cur_a;
                        m4_groups[3 * n_groups + 1] = (int32_t) r;
                        m4_groups[3 * n_groups + 2] = 8;
                        n_groups++;
                        offset += m8_grp_bytes;
                    }
                    if (pad8_all && r < cne1) {
                        m4_groups[3 * n_groups + 0] = cur_a;
                        m4_groups[3 * n_groups + 1] = (int32_t) r;
                        m4_groups[3 * n_groups + 2] = (int32_t) (cne1 - r);  // 1..7 real rows of an 8-row group
                        n_groups++;
                        offset += m8_grp_bytes;
                        r = cne1;
                    }
                    for (; r + 4 <= cne1; r += 4) {
                        m4_groups[3 * n_groups + 0] = cur_a;
                        m4_groups[3 * n_groups + 1] = (int32_t) r;
                        m4_groups[3 * n_groups + 2] = 4;
                        n_groups++;
                        offset += 4 * nbw1_m4;
                    }
                }
                m4_n_groups[0] = n_groups;
            }

            if (op_stats) {
                mmid_route_stats & r = g_mmid_route[st_kind];
                r.calls++;
                r.active_experts += valid_ep_count_t;
                r.rows += valid_act_count_t;
                for (int cur_a = 0; cur_a < n_as; ++cur_a) {
                    int64_t c = matrix_row_counts[cur_a];
                    if (c == 0) {
                        continue;
                    }
                    int b = 0;
                    while (c > 1 && b < 8) {
                        c >>= 1;
                        b++;
                    }
                    r.hist[b]++;
                }
                r.tcm_size = (int64_t) ctx.shared.size;
            }
            st_lap(op_stats ? st->t_route : st_t);
        }

        const int64_t barrier_idx = static_cast<int64_t>(ith / 2);

        GGML_ASSERT(global_spine_env_info.init_barrier != nullptr);
        GGML_ASSERT(barrier_idx < static_cast<int64_t>(spine_init_barrier_count));
        spine_barrier_t * cur_barrier = &global_spine_env_info.init_barrier[barrier_idx];

        ctx.sync();

        st_lap(op_stats ? st->t_sync : st_t);

        if (use_m4 && m4_n_groups[0] > 0 && !reuse_q) {
            const int64_t n_groups = m4_n_groups[0];
            for (int64_t gi = ith; gi < n_groups; gi += nth) {
                const int     cur_a = m4_groups[3 * gi + 0];
                const int64_t r0    = m4_groups[3 * gi + 1];
                const int     nr    = m4_groups[3 * gi + 2];
                for (int r = 0; r < nr; ++r) {
                    const mmid_row_mapping rm = MMID_MATRIX_ROW(cur_a, r0 + r);
                    const float * src = (const float *) ((const char *) src1->data + (rm.i1 % ne11) * nb11 + rm.i2 * nb12);
                    memcpy(m4_a_f32 + r * ne10, src, ne10 * sizeof(float));
                }
                // groups of an expert are laid out in row order: n8 x m8 group, then at most one m4 group
                const int64_t n8  = use_m8 ? std::min<int64_t>(r0, (matrix_row_counts[cur_a] / 8) * 8) / 8 : 0;
                uint8_t *     out = m4_quant + m4_expert_offset[cur_a] + n8 * m8_grp_bytes + (r0 - n8 * 8) * nbw1_m4;
                if (nr < 8 && pad8_all) {
                    memset(m4_a_f32 + nr * ne10, 0, (8 - nr) * ne10 * sizeof(float));
                    spacemit_kernels::rvv::quantize_a_8row_i8_hp_m8(a_blk_len, m4_a_f32, ne10, out);
                } else if (nr == 8) {
                    spacemit_kernels::rvv::quantize_a_8row_i8_hp_m8(a_blk_len, m4_a_f32, ne10, out);
                } else {
                    quantize_a_4row_i8(a_blk_len, m4_a_f32, ne10, out);
                }
            }
            st_lap(op_stats ? st->t_quant4 : st_t);
            ctx.sync();
            st_lap(op_stats ? st->t_q4wait : st_t);
        }

        const size_t row_stride_b      = b_k_blks * get_repacked_block_type_size<BLOC_TYPE, INTER_SIZE, NB_COLS>();
        const size_t expert_b_stride   = ne01 * row_stride_b;
        const size_t per_nb_cols_wsize = NB_COLS * row_stride_b;

        std::array<const uint8_t *, 2> src_workspaces;
        std::array<float *, 2>         dst_workspaces;

        const int64_t tcm_buffer_size = (int64_t) ctx.shared.size;
        void *        tcm_buffer      = ctx.shared.data;

        const auto valid_ep_count_t  = valid_ep_count[0];
        const auto valid_act_count_t = valid_act_count[0];

        int nth_es = 1;
        int nth_n  = nth;

        int ith_es = ith % nth_es;
        int ith_n  = (ith / nth_es) % nth_n;

        if (!pad8_all && valid_ep_count_t % nth == 0 && tcm_buffer != nullptr && valid_ep_count_t == n_as &&
            valid_act_count_t == n_as && per_nb_cols_wsize <= tcm_buffer_size) {
            const bool has_pair = ((ith & 1) != 0) || (ith + 1 < nth);

            if (op_stats) {
                st->fast_path++;
            }

            for (int64_t valid_id = ith; valid_id < valid_ep_count_t; valid_id += nth) {
                const int64_t cur_a = valid_matrix_row_counts[valid_id];

                auto * src0_cur = (uint8_t *) src0->data + cur_a * expert_b_stride;

                mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, 0);
                const int        id          = row_mapping.i1;
                const int64_t    i11         = id % ne11;
                const int64_t    i12         = row_mapping.i2;
                const int64_t    i1          = id;
                const int64_t    i2          = i12;

                auto *  src1_col = quant_a_buffer + (i11 * nbw1 + i12 * nbw2);
                float * c_blk    = (float *) ((char *) dst->data + (i1 * nb1 + i2 * nb2));

                uint8_t * a_row = src1_col;
                uint8_t * b_col = reinterpret_cast<uint8_t *>(tcm_buffer);
                if ((nbw1 + per_nb_cols_wsize) <= tcm_buffer_size) {
                    a_row = (uint8_t *) tcm_buffer;
                    b_col = reinterpret_cast<uint8_t *>(tcm_buffer) + nbw1;
                }
                uint8_t * b_col_zp = block_type_has_zp<BLOC_TYPE>() ? b_col : nullptr;

                if (ith % 2 == 0) {
                    spacemit_kernels::rvv::memcpy1d(b_col, reinterpret_cast<uint8_t *>(src0_cur), per_nb_cols_wsize);

                    if (a_row != src1_col) {
                        spacemit_kernels::rvv::memcpy1d(a_row, src1_col, nbw1);
                    }
                }

                if (has_pair) {
                    spine_barrier_wait(cur_barrier);
                }

                if (ith % 2 != 0) {
                    if (a_row != src1_col) {
                        spacemit_kernels::rvv::memcpy1d(a_row, src1_col, nbw1);
                    }

                    spacemit_kernels::rvv::memcpy1d(b_col, reinterpret_cast<uint8_t *>(src0_cur), per_nb_cols_wsize);
                }

                int64_t nb_real = std::min(ne01, NB_COLS);
                for (int64_t ni = 0; ni < ne01; ni += NB_COLS) {
                    if (has_pair && ith % 2 != 0) {
                        spine_barrier_wait(cur_barrier);
                    }

                    gemm_kernel(b_blk_len, a_row, b_col, b_col_zp, c_blk + ni, 1, nb_real, b_k_blks, ne01);
                    mmid_scale_row(c_blk + ni, nb_real, fw, i1, i2);

                    if (has_pair && ith % 2 == 0) {
                        spine_barrier_wait(cur_barrier);
                    }

                    const int64_t next_ni = ni + NB_COLS;
                    if (next_ni < ne01) {
                        nb_real = std::min(ne01 - next_ni, NB_COLS);
                        spacemit_kernels::rvv::memcpy1d(
                            b_col, reinterpret_cast<uint8_t *>(src0_cur) + next_ni * row_stride_b, per_nb_cols_wsize);
                    }
                }
            }
        } else {
            for (int64_t valid_id = ith_es; valid_id < valid_ep_count_t; valid_id += nth_es) {
                const int64_t cur_a = valid_matrix_row_counts[valid_id];
                const int64_t cne1  = matrix_row_counts[cur_a];

                int64_t src1_cur_start = 0;
                int64_t src1_cur_end   = cne1;

                int64_t src0_cur_start = (ith_n * ne01) / nth_n;
                int64_t src0_cur_end   = MIN(((ith_n + 1) * ne01) / nth_n, ne01);

                if (src1_cur_start >= src1_cur_end || src0_cur_start >= src0_cur_end) {
                    continue;
                }

                src0_cur_start =
                    (src0_cur_start % NB_COLS) ? src0_cur_start + NB_COLS - (src0_cur_start % NB_COLS) : src0_cur_start;
                src0_cur_end =
                    (src0_cur_end % NB_COLS) ? src0_cur_end + NB_COLS - (src0_cur_end % NB_COLS) : src0_cur_end;

                auto *    src0_cur = (uint8_t *) src0->data + cur_a * expert_b_stride + src0_cur_start * row_stride_b;
                uint8_t * b_col_zp = block_type_has_zp<BLOC_TYPE>() ? src0_cur : nullptr;

                size_t extra_tcm_buffer_size = tcm_buffer_size;
                void * extra_tcm_buffer      = tcm_buffer;
                if (op_stats) {
                    st_lap(st->t_kernel);  // bookkeeping between experts
                    st->kernel_flops += 2.0 * (double) ne00 * (double) (src0_cur_end - src0_cur_start) *
                                        (double) (src1_cur_end - src1_cur_start);
                }

                if (tcm_buffer != nullptr && (src1_cur_end - src1_cur_start) >= 4 &&
                    (src0_cur_end - src0_cur_start) * row_stride_b <= tcm_buffer_size) {
                    if (op_stats) {
                        st->w_tcm_hit++;
                    }
                    spacemit_kernels::rvv::memcpy1d(tcm_buffer, src0_cur,
                                                    (src0_cur_end - src0_cur_start) * row_stride_b);
                    src0_cur = reinterpret_cast<uint8_t *>(tcm_buffer);
                    b_col_zp = block_type_has_zp<BLOC_TYPE>() ? src0_cur : nullptr;
                    extra_tcm_buffer_size -= (src0_cur_end - src0_cur_start) * row_stride_b;
                    extra_tcm_buffer = reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(tcm_buffer) +
                                                                (src0_cur_end - src0_cur_start) * row_stride_b);
                    st_lap(op_stats ? st->t_wcopy : st_t);
                } else if (op_stats) {
                    st->w_tcm_miss++;
                }

                int ir1 = src1_cur_start;

                const int64_t n_cols = src0_cur_end - src0_cur_start;
                if (pad8_all) {
                    GGML_ASSERT(use_m8 && n_cols % 32 == 0);
                }
                if (use_m4 && n_cols % 32 == 0 && (cne1 >= 4 || pad8_all)) {
                    const uint8_t * a_grp = m4_quant + m4_expert_offset[cur_a];
                    st_lap(op_stats ? st->t_kernel : st_t);
                    for (; use_m8 && ir1 + 8 <= src1_cur_end; ir1 += 8, a_grp += m8_grp_bytes) {
                        spacemit_kernels::ime2::gemm_kernel_i8i4_hp_m8(b_blk_len, a_grp, src0_cur, nullptr, m4_c_tmp, 8,
                                                                       n_cols, b_k_blks, n_cols);
                        if (op_stats) {
                            st_lap(st->t_k8);
                            st->rows_m8 += 8;
                            st->flops_m8 += 2.0 * 8 * ne00 * n_cols;
                        }
                        for (int r = 0; r < 8; ++r) {
                            const mmid_row_mapping rm = MMID_MATRIX_ROW(cur_a, ir1 + r);
                            float * dst_row = (float *) ((char *) dst->data + (rm.i1 * nb1 + rm.i2 * nb2)) + src0_cur_start;
                            mmid_store_row(dst_row, m4_c_tmp + r * n_cols, n_cols, fw, rm.i1, rm.i2);
                        }
                        if (op_stats && ne01 > 1024) {
                            const int64_t t_before = st->t_scatter;
                            st_lap(st->t_scatter);
                            st->t_scatter_wide += st->t_scatter - t_before;
                        } else {
                            st_lap(op_stats ? st->t_scatter : st_t);
                        }
                    }
                    if (pad8_all && ir1 < src1_cur_end) {
                        // zero-padded tail group: compute 8 rows, keep the real ones
                        const int nr = (int) (src1_cur_end - ir1);
                        spacemit_kernels::ime2::gemm_kernel_i8i4_hp_m8(b_blk_len, a_grp, src0_cur, nullptr, m4_c_tmp, 8,
                                                                       n_cols, b_k_blks, n_cols);
                        if (op_stats) {
                            st_lap(st->t_k8);
                            st->rows_m8 += nr;
                            st->flops_m8 += 2.0 * 8 * ne00 * n_cols;
                        }
                        for (int r = 0; r < nr; ++r) {
                            const mmid_row_mapping rm = MMID_MATRIX_ROW(cur_a, ir1 + r);
                            float * dst_row = (float *) ((char *) dst->data + (rm.i1 * nb1 + rm.i2 * nb2)) + src0_cur_start;
                            mmid_store_row(dst_row, m4_c_tmp + r * n_cols, n_cols, fw, rm.i1, rm.i2);
                        }
                        st_lap(op_stats ? st->t_scatter : st_t);
                        ir1 = src1_cur_end;
                    }
                    for (; ir1 + 4 <= src1_cur_end; ir1 += 4, a_grp += 4 * nbw1_m4) {
                        gemm_kernel(b_blk_len, a_grp, src0_cur, b_col_zp, m4_c_tmp, 4, n_cols, b_k_blks, n_cols);
                        if (op_stats) {
                            st_lap(st->t_k4);
                            st->flops_m4 += 2.0 * 4 * ne00 * n_cols;
                        }
                        for (int r = 0; r < 4; ++r) {
                            const mmid_row_mapping rm = MMID_MATRIX_ROW(cur_a, ir1 + r);
                            float * dst_row = (float *) ((char *) dst->data + (rm.i1 * nb1 + rm.i2 * nb2)) + src0_cur_start;
                            mmid_store_row(dst_row, m4_c_tmp + r * n_cols, n_cols, fw, rm.i1, rm.i2);
                        }
                        if (op_stats && ne01 > 1024) {
                            const int64_t t_before = st->t_scatter;
                            st_lap(st->t_scatter);
                            st->t_scatter_wide += st->t_scatter - t_before;
                        } else {
                            st_lap(op_stats ? st->t_scatter : st_t);
                        }
                    }
                    if (op_stats) {
                        st->rows_m4 += (cne1 / 4) * 4;
                    }
                    st_lap(op_stats ? st->t_kernel : st_t);
                }

                if (extra_tcm_buffer_size >= nbw1 && extra_tcm_buffer != nullptr) {
                    int64_t quant_a_tile_size = extra_tcm_buffer_size / nbw1;
                    do {
                        quant_a_tile_size = MIN(quant_a_tile_size, src1_cur_end - ir1);

                        uint8_t * quant_a_tile_buffer = reinterpret_cast<uint8_t *>(extra_tcm_buffer);

                        int iir1 = ir1;
                        for (; iir1 < (ir1 + quant_a_tile_size); ++iir1) {
                            mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, iir1);

                            const int id = row_mapping.i1;  // selected expert index

                            const int64_t i11 = id % ne11;
                            const int64_t i12 = row_mapping.i2;  // row index in src1

                            auto * src1_col = quant_a_buffer + (i11 * nbw1 + i12 * nbw2);
                            spacemit_kernels::rvv::memcpy1d(quant_a_tile_buffer, src1_col, nbw1);
                            quant_a_tile_buffer = quant_a_tile_buffer + nbw1;
                        }

                        st_lap(op_stats ? st->t_agather : st_t);

                        quant_a_tile_buffer = reinterpret_cast<uint8_t *>(extra_tcm_buffer);
                        iir1                = ir1;

                        if (moe_gemm_kernel_m2 != nullptr) {
                            for (; iir1 < (ir1 + quant_a_tile_size - 1); iir1 += 2, quant_a_tile_buffer += 2 * nbw1) {
                                mmid_row_mapping row_mapping_0 = MMID_MATRIX_ROW(cur_a, iir1);
                                mmid_row_mapping row_mapping_1 = MMID_MATRIX_ROW(cur_a, iir1 + 1);

                                src_workspaces[0] = quant_a_tile_buffer;
                                src_workspaces[1] = quant_a_tile_buffer + nbw1;

                                dst_workspaces[0] =
                                    (float *) ((char *) dst->data + (row_mapping_0.i1 * nb1 + row_mapping_0.i2 * nb2)) +
                                    src0_cur_start;
                                dst_workspaces[1] = (float *) ((char *) dst->data +
                                                               ((row_mapping_1.i1) * nb1 + (row_mapping_1.i2) * nb2)) +
                                                    src0_cur_start;
                                moe_gemm_kernel_m2(b_blk_len, src_workspaces.data(), src0_cur, b_col_zp,
                                                   dst_workspaces.data(), 1, src0_cur_end - src0_cur_start, b_k_blks,
                                                   ne01);
                                mmid_scale_row(dst_workspaces[0], src0_cur_end - src0_cur_start, fw, row_mapping_0.i1,
                                               row_mapping_0.i2);
                                mmid_scale_row(dst_workspaces[1], src0_cur_end - src0_cur_start, fw, row_mapping_1.i1,
                                               row_mapping_1.i2);
                                if (op_stats) {
                                    st->rows_m2 += 2;
                                }
                            }
                        }

                        for (; iir1 < (ir1 + quant_a_tile_size); iir1++, quant_a_tile_buffer += nbw1) {
                            mmid_row_mapping row_mapping_0 = MMID_MATRIX_ROW(cur_a, iir1);
                            if (op_stats) {
                                st->rows_m1++;
                            }

                            float * y = (float *) ((char *) dst->data + (row_mapping_0.i1 * nb1 + row_mapping_0.i2 * nb2)) +
                                        src0_cur_start;
                            gemm_kernel(b_blk_len, quant_a_tile_buffer, src0_cur, b_col_zp, y, 1,
                                        src0_cur_end - src0_cur_start, b_k_blks, ne01);
                            mmid_scale_row(y, src0_cur_end - src0_cur_start, fw, row_mapping_0.i1, row_mapping_0.i2);
                        }

                        st_lap(op_stats ? st->t_kernel : st_t);

                        ir1 += quant_a_tile_size;
                    } while (ir1 < src1_cur_end);
                } else {
                    if (moe_gemm_kernel_m2 != nullptr) {
                        for (; ir1 < src1_cur_end - 1; ir1 += 2) {
                            for (int iir1 = 0; iir1 < 2; ++iir1) {
                                mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, ir1 + iir1);

                                const int id = row_mapping.i1;  // selected expert index

                                const int64_t i11 = id % ne11;
                                const int64_t i12 = row_mapping.i2;  // row index in src1

                                const int64_t i1 = id;               // selected expert index
                                const int64_t i2 = i12;              // row

                                src_workspaces[iir1] = quant_a_buffer + (i11 * nbw1 + i12 * nbw2);

                                dst_workspaces[iir1] =
                                    (float *) ((char *) dst->data + (i1 * nb1 + i2 * nb2)) + src0_cur_start;
                            }

                            moe_gemm_kernel_m2(b_blk_len, src_workspaces.data(), src0_cur, b_col_zp,
                                               dst_workspaces.data(), 1, src0_cur_end - src0_cur_start, b_k_blks, ne01);
                            if (fw != nullptr) {
                                for (int iir1 = 0; iir1 < 2; ++iir1) {
                                    const mmid_row_mapping rm = MMID_MATRIX_ROW(cur_a, ir1 + iir1);
                                    mmid_scale_row(dst_workspaces[iir1], src0_cur_end - src0_cur_start, fw, rm.i1,
                                                   rm.i2);
                                }
                            }
                            if (op_stats) {
                                st->rows_m2 += 2;
                            }
                        }
                    }

                    for (; ir1 < src1_cur_end; ir1++) {
                        if (op_stats) {
                            st->rows_m1++;
                        }
                        mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, ir1);

                        const int id = row_mapping.i1;  // selected expert index

                        const int64_t i11 = id % ne11;
                        const int64_t i12 = row_mapping.i2;  // row index in src1

                        const int64_t i1 = id;               // selected expert index
                        const int64_t i2 = i12;              // row

                        auto * src1_col = quant_a_buffer + (i11 * nbw1 + i12 * nbw2);

                        float * y = (float *) ((char *) dst->data + (i1 * nb1 + i2 * nb2)) + src0_cur_start;
                        gemm_kernel(b_blk_len, src1_col, src0_cur, b_col_zp, y, 1, src0_cur_end - src0_cur_start,
                                    b_k_blks, ne01);
                        mmid_scale_row(y, src0_cur_end - src0_cur_start, fw, i1, i2);
                    }
                    st_lap(op_stats ? st->t_kernel : st_t);
                }
            }
        }

        if (op_stats) {
            st_lap(st->t_kernel);  // fast path and trailing bookkeeping
            st->t_total += st_t - st_t0;
        }
#undef MMID_MATRIX_ROW
    }

    int repack(ggml_tensor * t, const void * data, size_t data_size) const override {
        GGML_LOG_DEBUG("%s: repack tensor %s with %s_%dx%d\n", __func__, t->name, ggml_type_name(t->type),
                       (int) NB_COLS, (int) INTER_SIZE);
        return ggml::cpu::riscv64_spacemit::repack<BLOC_TYPE, INTER_SIZE, NB_COLS>(t, data, data_size);
    }
};

class tensor_traits_common : public ggml::spacemit::tensor_traits_base {
    bool work_size(int n_threads, const ggml_tensor * op, size_t & size) const override {
        switch (op->op) {
            case GGML_OP_MUL_MAT:
                if (op->src[0] && op->src[1] && op->src[1]->type == GGML_TYPE_F32) {
                    const auto * traits = ggml_get_type_traits_cpu(op->src[0]->type);
                    if (traits && traits->vec_dot) {
                    const size_t row_size = ggml_row_size(traits->vec_dot_type, op->src[1]->ne[0]);
                    const size_t bytes = row_size * op->src[1]->ne[1] * op->src[1]->ne[2] * op->src[1]->ne[3];
                        if (op->src[1]->type != traits->vec_dot_type) {
                            size = bytes;
                            return true;
                        }
                    }
                }
                break;
            case GGML_OP_FLASH_ATTN_EXT:
                {
                    const int     n_tasks = n_threads;
                    const int64_t neq2    = op->src[0]->ne[2];  // number of query heads
                    const int64_t DK      = op->src[1]->ne[0];
                    const int64_t DV      = op->src[2]->ne[0];  // DV

                    // Tiled flash attention scratch (tile sizes defined in common.h)
                    // Per-thread: Q_q + KQ + mask + VKQ32 + V32 + K_f32 + padding
                    size_t prefill = sizeof(float) *
                                     (GGML_FA_TILE_Q * DK + 2 * GGML_FA_TILE_Q * GGML_FA_TILE_KV + GGML_FA_TILE_Q * DV +
                                      GGML_FA_TILE_KV * DV + GGML_FA_TILE_KV * DK +
                                      ggml::spacemit::cache_line_size_f32) *
                                     n_tasks;

                    // Decode path: n_kv_chunks = n_tasks (one chunk per thread)
                    // Per-thread: VKQ accmulator (DV), partial M, partial S + intra-thread scratch for V, Q and VKQ
                    size_t n_chunks = n_tasks;
                    size_t decode   = sizeof(float) * n_tasks * (DK + 2 * DV + 64);
                    // split-KV decode partials (up to 4 query tokens, FA_SPLIT_KV_MAX_Q)
                    const size_t n_q = (size_t) std::min<int64_t>(op->src[0]->ne[1], 4);
                    decode = MAX(decode, sizeof(float) * (n_q * neq2 * n_chunks * (2 + DV) + n_tasks * (DK + 2 * DV)));

                    size = MAX(prefill, decode);
                }
                return true;
            case GGML_OP_ROPE:
                {
                    // forward_rope_impl caches sin/cos per thread unless the
                    // small-F32-NEOX legacy path (stack cache) is taken
                    const int  mode    = ggml_get_op_params_i32(op, 2);
                    const bool legacy  = op->src[0]->type == GGML_TYPE_F32 &&
                                        mode == GGML_ROPE_TYPE_NEOX && op->ne[0] <= 128 &&
                                        op->src[0]->nb[0] == sizeof(float);
                    size = legacy ? 0
                                  : (size_t) n_threads *
                                        (op->ne[0] + ggml::spacemit::cache_line_size_f32) * sizeof(float);
                }
                return true;
            default:
                break;
        }
        return false;
    }

    bool compute_forward(ggml::spacemit::context & ctx, ggml_tensor * op) const override {
        switch (op->op) {
            case GGML_OP_MUL_MAT:
                if (op->src[0]->type == GGML_TYPE_F16 &&
                    op->src[1]->type == GGML_TYPE_F32 &&
                    op->src[0]->nb[0] == sizeof(_Float16) &&
                    op->src[1]->nb[0] == sizeof(float)) {
                    spacemit_kernels::rvv::forward_mul_mat_f16_f32(ctx, op);
                    return true;
                }
                if (op->src[1]->type == GGML_TYPE_F32 &&
                    ggml_get_type_traits_cpu(op->src[0]->type)->vec_dot != nullptr) {
                    spacemit_kernels::scalar::forward_mul_mat(ctx, op);
                    return true;
                }
                return false;
            case GGML_OP_NORM:
                switch (op->src[0]->type) {
                    case GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_norm_f32(ctx, op);
                        return true;
                    case GGML_TYPE_F16:
                        spacemit_kernels::scalar::forward_norm_f16(ctx, op);
                        return true;
                    default:
                        GGML_ABORT("fatal error");
                }
            case GGML_OP_RMS_NORM:
                switch (op->src[0]->type) {
                    case GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_rms_norm_f32(ctx, op);
                        return true;
                    case GGML_TYPE_F16:
                        spacemit_kernels::scalar::forward_rms_norm_f16(ctx, op);
                        return true;
                    default:
                        GGML_ABORT("fatal error");
                }
            case GGML_OP_ADD:
                switch (op->src[0]->type) {
                    case GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_binary<GGML_OP_ADD, float>(ctx, op);
                        return true;
                    case GGML_TYPE_F16:
                        spacemit_kernels::rvv::forward_binary<GGML_OP_ADD, _Float16>(ctx, op);
                        return true;
                    default:
                        return false;
                }
            case GGML_OP_SUB:
                switch (op->src[0]->type) {
                    case GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_binary<GGML_OP_SUB, float>(ctx, op);
                        return true;
                    case GGML_TYPE_F16:
                        spacemit_kernels::rvv::forward_binary<GGML_OP_SUB, _Float16>(ctx, op);
                        return true;
                    default:
                        return false;
                }
            case GGML_OP_MUL:
                if (mul_is_fused_moe_w(op)) {
                    // src0 already holds the weighted rows (MUL_MAT_ID applied them, see graph_prepare_fusions)
                    if (op->data != op->src[0]->data) {
                        const size_t n  = ggml_nbytes(op);
                        const size_t dn = (n / ctx.nth + 63) & ~(size_t) 63;
                        const size_t b0 = std::min(n, dn * ctx.ith);
                        const size_t b1 = std::min(n, b0 + dn);
                        if (b1 > b0) {
                            memcpy((char *) op->data + b0, (const char *) op->src[0]->data + b0, b1 - b0);
                        }
                    }
                    return true;
                }
                switch (op->src[0]->type) {
                    case GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_binary<GGML_OP_MUL, float>(ctx, op);
                        return true;
                    case GGML_TYPE_F16:
                        spacemit_kernels::rvv::forward_binary<GGML_OP_MUL, _Float16>(ctx, op);
                        return true;
                    default:
                        return false;
                }
            case GGML_OP_DIV:
                switch (op->src[0]->type) {
                    case GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_binary<GGML_OP_DIV, float>(ctx, op);
                        return true;
                    case GGML_TYPE_F16:
                        spacemit_kernels::rvv::forward_binary<GGML_OP_DIV, _Float16>(ctx, op);
                        return true;
                    default:
                        return false;
                }
            case GGML_OP_SCALE:
                if (op->src[0]->type == GGML_TYPE_F32) {
                    spacemit_kernels::rvv::forward_scale_f32(ctx, op);
                    return true;
                }
                return false;
            case GGML_OP_UNARY:
                switch (ggml_get_unary_op(op)) {
                    case GGML_UNARY_OP_TANH:
                        spacemit_kernels::rvv::forward_unary_tanh_f32(ctx, op);
                        return true;
                    case GGML_UNARY_OP_GELU:
                        spacemit_kernels::rvv::forward_unary_gelu_f32(ctx, op);
                        return true;
                    case GGML_UNARY_OP_SILU:
                        spacemit_kernels::rvv::forward_unary_silu_f32(ctx, op);
                        return true;
                    case GGML_UNARY_OP_EXP:
                        spacemit_kernels::rvv::forward_unary_exp_f32(ctx, op);
                        return true;
                    case GGML_UNARY_OP_SIGMOID:
                        spacemit_kernels::rvv::forward_unary_sigmoid_f32(ctx, op);
                        return true;
                    case GGML_UNARY_OP_NEG:
                        spacemit_kernels::rvv::forward_unary_neg_f32(ctx, op);
                        return true;
                    case GGML_UNARY_OP_SOFTPLUS:
                        spacemit_kernels::rvv::forward_unary_softplus_f32(ctx, op);
                        return true;
                    default:
                        return false;
                }
            case GGML_OP_SOFT_MAX:
                if (op->src[0]->type == GGML_TYPE_F32) {
                    spacemit_kernels::rvv::forward_soft_max_f32(ctx, op);
                    return true;
                }
                return false;
            case GGML_OP_GLU:
                if (op->src[0]->type == GGML_TYPE_F32) {
                    switch (ggml_get_glu_op(op)) {
                        case GGML_GLU_OP_GEGLU:
                            spacemit_kernels::rvv::forward_glu_geglu_f32(ctx, op);
                            return true;
                        case GGML_GLU_OP_SWIGLU:
                            spacemit_kernels::rvv::forward_glu_swiglu_f32(ctx, op);
                            return true;
                        default:
                            break;
                    }
                }
                return false;
            case GGML_OP_FLASH_ATTN_EXT:
                // Use the RVV implementation for its supported F32-Q/F16-KV
                // shapes; retain the CPU-equivalent implementation for all
                // other layouts and dtypes.
                if (forward_flash_attn_ext_f16(ctx, op)) {
                    return true;
                }
                spacemit_kernels::scalar::forward_flash_attn_ext(ctx, op);
                return true;
            case GGML_OP_ROPE:
                spacemit_kernels::rvv::forward_rope(ctx, op);
                return true;
            case GGML_OP_CONT:
                {
                    const ggml_tensor * src0 = op->src[0];
                    if (op->type == src0->type && op->nb[0] != src0->nb[0] && op->nb[0] == src0->nb[1] &&
                        op->ne[3] * op->ne[2] * op->nb[2] == src0->ne[3] * src0->ne[2] * src0->nb[2]) {
                        spacemit_kernels::rvv::forward_cont_with_permute(ctx, op);
                        return true;
                    }
                    if (op->type == src0->type && ggml_is_contiguous(op) &&
                        ggml_nelements(op) == ggml_nelements(src0) &&
                        (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16)) {
                        if (op->type == GGML_TYPE_F32) {
                            spacemit_kernels::rvv::forward_cont_general<float>(ctx, op);
                        } else {
                            spacemit_kernels::rvv::forward_cont_general<_Float16>(ctx, op);
                        }
                        return true;
                    }
                    return false;
                }
            case GGML_OP_CPY:
                {
                    const ggml_tensor * src0 = op->src[0];
                    if (ggml_nelements(op) == 0) {
                        return true;
                    }
                    if (op->type == src0->type && op->nb[0] == src0->nb[1] && src0->nb[0] != src0->nb[1] &&
                        ggml_nelements(src0) == ggml_nelements(op)) {
                        spacemit_kernels::rvv::forward_cpy_with_permute(ctx, op);
                        return true;
                    }
                    if (op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32 &&
                        ggml_is_contiguous(op) && ggml_nelements(src0) == ggml_nelements(op)) {
                        spacemit_kernels::rvv::forward_cpy_strided_f32(ctx, op);
                        return true;
                    }
                    if (ggml_is_contiguous(op) && ggml_is_contiguous(src0) &&
                        ggml_nelements(src0) == ggml_nelements(op)) {
                        if (op->type == GGML_TYPE_F16 && src0->type == GGML_TYPE_F32) {
                            spacemit_kernels::rvv::forward_cpy_f32_to_f16(ctx, op);
                            return true;
                        }
                        if (op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F16) {
                            spacemit_kernels::rvv::forward_cpy_f16_to_f32(ctx, op);
                            return true;
                        }
                    }
                    return false;
                }
            case GGML_OP_SET_ROWS:
                spacemit_kernels::rvv::forward_set_rows(ctx, op);
                return true;
            case GGML_OP_REPEAT:
                {
                    const bool rows_equal         = ggml_nrows(op->src[0]) == ggml_nrows(op);
                    const bool broadcast_or_equal = op->src[0]->ne[0] == 1 || op->src[0]->ne[0] == op->ne[0];

                    if (rows_equal && broadcast_or_equal) {
                        switch (op->src[0]->type) {
                            case GGML_TYPE_F32:
                                spacemit_kernels::rvv::forward_repeat_nrows<int32_t>(ctx, op);
                                return true;
                            case GGML_TYPE_F16:
                                spacemit_kernels::rvv::forward_repeat_nrows<int16_t>(ctx, op);
                                return true;
                            default:
                                break;
                        }
                    }

                    if (op->src[0]->ne[1] == 1 && op->src[0]->ne[0] == op->ne[0]) {
                        switch (op->src[0]->type) {
                            case GGML_TYPE_F32:
                                spacemit_kernels::rvv::forward_repeat_dim1<int32_t>(ctx, op);
                                return true;
                            case GGML_TYPE_F16:
                                spacemit_kernels::rvv::forward_repeat_dim1<int16_t>(ctx, op);
                                return true;
                            default:
                                break;
                        }
                    }

                    return false;
                }
                return true;
            case GGML_OP_SUM_ROWS:
                {
                    if (op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32) {
                        spacemit_kernels::rvv::forward_sum_rows<float>(ctx, op);
                    } else {
                        return false;
                    }
                }
                return true;
            case GGML_OP_GET_ROWS:
                {
                    if (op->src[0]->type == op->type) {
                        switch (op->src[0]->type) {
                            case GGML_TYPE_F32:
                                spacemit_kernels::rvv::forward_get_rows<int32_t>(ctx, op);
                                return true;
                            case GGML_TYPE_F16:
                                spacemit_kernels::rvv::forward_get_rows<int16_t>(ctx, op);
                                return true;
                            default:
                                break;
                        }
                    }

                    return false;
                }
                return true;
            case GGML_OP_CONCAT:
                {
                    // fused with the ssm_conv: only the trailing columns read back through views are written
                    const int32_t fused_col = ggml_get_op_params_i32(op, FUSE_P_STATE);
                    if (spacemit_kernels::rvv::forward_concat_dim0_transposed_f32(ctx, op, fused_col > 0 ? fused_col - 1 : 0)) {
                        return true;
                    }
                    const int32_t dim = ggml_get_op_params_i32(op, 0);
                    if (dim == 0 && op->type == op->src[0]->type) {
                        switch (op->src[0]->type) {
                            case GGML_TYPE_F32:
                                spacemit_kernels::rvv::forward_concat<int32_t>(ctx, op);
                                return true;
                            case GGML_TYPE_F16:
                                spacemit_kernels::rvv::forward_concat<int16_t>(ctx, op);
                                return true;
                            default:
                                break;
                        }
                    }

                    return false;
                }
                return true;
            // Scalar ops dispatched to scalar_kernels
            case GGML_OP_L2_NORM:
                spacemit_kernels::scalar::forward_l2_norm_f32(ctx, op);
                return true;
            case GGML_OP_FILL:
                spacemit_kernels::scalar::forward_fill_f32(ctx, op);
                return true;
            case GGML_OP_CUMSUM:
                spacemit_kernels::scalar::forward_cumsum_f32(ctx, op);
                return true;
            case GGML_OP_PAD:
                spacemit_kernels::scalar::forward_pad_f32(ctx, op);
                return true;
            case GGML_OP_TRI:
                spacemit_kernels::scalar::forward_tri_f32(ctx, op);
                return true;
            case GGML_OP_DIAG:
                spacemit_kernels::scalar::forward_diag_f32(ctx, op);
                return true;
            case GGML_OP_SET:
                spacemit_kernels::scalar::forward_set_f32(ctx, op);
                return true;
            case GGML_OP_SOLVE_TRI:
                spacemit_kernels::scalar::forward_solve_tri_f32(ctx, op);
                return true;
            case GGML_OP_GATED_DELTA_NET:
                spacemit_kernels::rvv::forward_gated_delta_net(ctx, op);
                return true;
            case GGML_OP_ARGSORT:
                spacemit_kernels::scalar::forward_argsort_f32(ctx, op);
                return true;
            case GGML_OP_CLAMP:
                spacemit_kernels::scalar::forward_clamp_f32(ctx, op);
                return true;
            case GGML_OP_SSM_CONV:
                if (op->src[0]->op == GGML_OP_CONCAT && ggml_get_op_params_i32(op->src[0], FUSE_P_STATE) > 0) {
                    // the concat only materialized its trailing columns; read its inputs directly
                    if (!spacemit_kernels::rvv::forward_ssm_conv_concat_f32(ctx, op)) {
                        GGML_ABORT("fused ssm_conv: unsupported shape");
                    }
                    return true;
                }
                if (!spacemit_kernels::rvv::forward_ssm_conv_f32(ctx, op)) {
                    spacemit_kernels::scalar::forward_ssm_conv_f32(ctx, op);
                }
                return true;
            default:
                break;
        }
        return false;
    }

    bool forward_flash_attn_ext_f16(ggml::spacemit::context & ctx, ggml_tensor * dst) const {
        const ggml_tensor * q = dst->src[0];
        const ggml_tensor * k = dst->src[1];
        const ggml_tensor * v = dst->src[2];

        GGML_TENSOR_LOCALS(int64_t, neq, q, ne)
        GGML_TENSOR_LOCALS(size_t, nbq, q, nb)
        GGML_TENSOR_LOCALS(int64_t, nek, k, ne)
        GGML_TENSOR_LOCALS(size_t, nbk, k, nb)
        GGML_TENSOR_LOCALS(int64_t, nev, v, ne)
        GGML_TENSOR_LOCALS(size_t, nbv, v, nb)
        GGML_TENSOR_LOCALS(int64_t, ne, dst, ne)
        GGML_TENSOR_LOCALS(size_t, nb, dst, nb)

        const int64_t DK = nek0;
        const int64_t DV = nev0;

        const bool supported_prec  = (dst->op_params[3] == GGML_PREC_F32 || dst->op_params[3] == GGML_PREC_DEFAULT);
        const bool supported_types = (q->type == GGML_TYPE_F32 && k->type == GGML_TYPE_F16 && v->type == GGML_TYPE_F16);
        const bool supported_shape = (DK > 0 && DK <= 256 && DV > 0 && DV <= 256);
        const bool supported_vlen  = (__riscv_vlenb() == 128);

        if (!(supported_prec && supported_types && supported_shape && supported_vlen)) {
            return false;
        }

        // total rows in q
        const int64_t nr = neq1 * neq2 * neq3;

        // rows per thread
        const int ith = ctx.ith;
        const int nth = ctx.nth;

        static constexpr int64_t Q_TILE_SZ = ggml_fa_tile_config::Q;
        const bool               use_tiled = neq1 >= Q_TILE_SZ;

        // 4x chunks per thread
        // int     nth_scaled = nth * 4;
        // int64_t chunk_size = (nr + nth_scaled - 1) / nth_scaled;
        // int64_t nchunk     = (nr + chunk_size - 1) / chunk_size;

        // if (nth == 1 || nchunk < nth) {
        //     nchunk = nth;
        // }

        // decode / MTP verification (1-4 query tokens): split the KV range across threads so each K/V row is
        // read once, then merge the partials (GGML_SPACEMIT_FA_SPLITKV=0 disables; ported from ggml-cpu/spacemit)
        static const bool split_kv = [] {
            const char * e = getenv("GGML_SPACEMIT_FA_SPLITKV");
            return !(e && e[0] == '0');
        }();
        const int n_split = (split_kv && !use_tiled && ctx.shared.data != nullptr) ?
                                spacemit_kernels::rvv::flash_attn_ext_f16_split_kv_chunks_vlen1024_vf16(dst, nth) :
                                0;
        if (n_split > 0) {
            float * partials = (float *) ctx.workspace;
            spacemit_kernels::rvv::forward_flash_attn_ext_f16_split_kv_partial_vlen1024_vf16(
                ctx, dst, n_split, partials, ctx.shared.data, ctx.shared.size);
            ctx.sync();
            spacemit_kernels::rvv::forward_flash_attn_ext_f16_split_kv_reduce_vlen1024_vf16(ctx, dst, n_split, partials);
            return true;
        }

        int64_t nchunk = nth;

        // The number of elements in each chunk
        const int64_t dr = (nr + nchunk - 1) / nchunk;

        // The first chunk comes from our thread_id, the rest will get auto-assigned.
        int current_chunk = ith;

        while (current_chunk < nchunk) {
            const int64_t ir0 = dr * current_chunk;
            const int64_t ir1 = MIN(ir0 + dr, nr);

            if (use_tiled) {
                spacemit_kernels::rvv::forward_flash_attn_ext_f16_tiled_vlen1024_vf16(
                    ctx, dst, ir0, ir1);
            } else {
                spacemit_kernels::rvv::forward_flash_attn_ext_f16_one_chunk_vlen1024_vf16(
                    ctx, dst, ir0, ir1);
            }

            current_chunk += nth;
        }

        return true;
    }

    int repack(ggml_tensor * t, const void * data, size_t data_size) const override {
        memcpy(t->data, data, data_size);
        return 0;
    }
};

// Impl By IME1
static const tensor_traits<block_q4_0, 32, 16>  q4_0_16x32_q8_0;
static const tensor_traits<block_q4_1, 32, 16>  q4_1_16x32_q8_0;
static const tensor_traits<block_q4_K, 32, 16>  q4_k_16x32_q8_0;
// Impl By IME2
static const tensor_traits<block_q2_K, 256, 32> q2_k_32x256_q8_0;
static const tensor_traits<block_q3_K, 256, 32> q3_k_32x256_q8_0;
static const tensor_traits<block_q4_0, 32, 32>  q4_0_32x32_q8_0;
static const tensor_traits<block_q4_1, 32, 32>  q4_1_32x32_q8_0;
static const tensor_traits<block_q4_0, 256, 32> q4_0_32x256_q8_0;
static const tensor_traits<block_q4_1, 256, 32> q4_1_32x256_q8_0;
static const tensor_traits<block_q4_K, 32, 32>  q4_k_32x32_q8_0;
static const tensor_traits<block_q6_K, 32, 32>  q6_k_32x32_q8_0;
static const tensor_traits<block_q8_0, 32, 32>  q8_0_32x32_q8_0;
static const tensor_traits<block_mxfp4, 32, 32> mxfp4_32x32_q8_0;
static const tensor_traits<block_q5_K, 32, 32>  q5_k_32x32_q8_0;
static const tensor_traits<block_q5_1, 32, 32>  q5_1_32x32_q8_0;
static const tensor_traits<block_q5_0, 32, 32>  q5_0_32x32_q8_0;

// Q5_0 weights converted exactly to Q8_0 at load time and run on the Q8_0 IME path (the Q5_0 IME kernel is
// opt-in: GGML_SPACEMIT_Q5_0_IME=1)
class tensor_traits_q5_0_as_q8_0 : public tensor_traits<block_q8_0, 32, 32> {
    int repack(ggml_tensor * t, const void * data, size_t data_size) const override {
        return repack_q5_0_to_q8_0_32(t, data, data_size);
    }
};

static const tensor_traits_q5_0_as_q8_0 q5_0_as_q8_0_32x32;

// F32 MoE router weights (ffn_gate_inp) quantized to Q8_0 at load and run on the Q8_0 IME path
// (GGML_SPACEMIT_ROUTER_Q8=0 disables; lossy: routing logits from Q8_0 weights)
class tensor_traits_f32_as_q8_0 : public tensor_traits<block_q8_0, 32, 32> {
    int repack(ggml_tensor * t, const void * data, size_t data_size) const override {
        return repack_f32_to_q8_0_32(t, data, data_size);
    }
};

static const tensor_traits_f32_as_q8_0 f32_as_q8_0_32x32;

// Q6_K lm_head (output.weight) requantized to Q4_0 at load (GGML_SPACEMIT_LM_Q4=1; lossy)
class tensor_traits_q6_k_as_q4_0 : public tensor_traits<block_q4_0, 256, 32> {
    int repack(ggml_tensor * t, const void * data, size_t data_size) const override {
        return repack_q6_k_to_q4_0_256_32(t, data, data_size);
    }
};

static const tensor_traits_q6_k_as_q4_0 q6_k_as_q4_0_32x256;
// Impl By RVV
static const tensor_traits_common               rvv_impl;

}  // namespace ggml::cpu::riscv64_spacemit

const ggml::spacemit::tensor_traits_base * ggml_spacemit_get_optimal_repack_type(const ggml_tensor * cur) {
    switch (cur->type) {
        case GGML_TYPE_Q2_K:
            {
#if defined(RISCV64_SPACEMIT_IME2)
                if (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    return &ggml::cpu::riscv64_spacemit::q2_k_32x256_q8_0;
                }
#endif
            }
            break;
        case GGML_TYPE_Q3_K:
            {
#if defined(RISCV64_SPACEMIT_IME2)
                if (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    return &ggml::cpu::riscv64_spacemit::q3_k_32x256_q8_0;
                }
#endif
            }
            break;
        case GGML_TYPE_Q4_0:
            {
#if defined(RISCV64_SPACEMIT_IME2)
                if (cur->ne[1] % 32 == 0 && cur->ne[0] % 256 == 0 &&
                    (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    return &ggml::cpu::riscv64_spacemit::q4_0_32x256_q8_0;
                }

                if (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    return &ggml::cpu::riscv64_spacemit::q4_0_32x32_q8_0;
                }
#endif

#if defined(RISCV64_SPACEMIT_IME1)
                if (cur->ne[1] % 16 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime1)) {
                    return &ggml::cpu::riscv64_spacemit::q4_0_16x32_q8_0;
                }
#endif
            }
            break;
        case GGML_TYPE_Q4_1:
            {
#if defined(RISCV64_SPACEMIT_IME2)
                // TODO
                // if (cur->ne[1] % 32 == 0 && cur->ne[0] % 256 == 0 &&
                //     (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                //     return &ggml::cpu::riscv64_spacemit::q4_1_32x256_q8_0;
                // }

                if (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    return &ggml::cpu::riscv64_spacemit::q4_1_32x32_q8_0;
                }
#endif

#if defined(RISCV64_SPACEMIT_IME1)
                if (cur->ne[1] % 16 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime1)) {
                    return &ggml::cpu::riscv64_spacemit::q4_1_16x32_q8_0;
                }
#endif
            }
            break;
        case GGML_TYPE_Q4_K:
            {
#if defined(RISCV64_SPACEMIT_IME2)
                if (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    return &ggml::cpu::riscv64_spacemit::q4_k_32x32_q8_0;
                }
#endif

#if defined(RISCV64_SPACEMIT_IME1)
                if (cur->ne[1] % 16 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime1)) {
                    return &ggml::cpu::riscv64_spacemit::q4_k_16x32_q8_0;
                }
#endif
            }
            break;
        case GGML_TYPE_Q6_K:
            {
#if defined(RISCV64_SPACEMIT_IME2)
                static const bool lm_q4 = [] {
                    const char * e = getenv("GGML_SPACEMIT_LM_Q4");
                    return e && e[0] == '1';
                }();
                if (lm_q4 && strcmp(cur->name, "output.weight") == 0 && ggml_n_dims(cur) == 2 && cur->ne[1] % 32 == 0 &&
                    cur->ne[0] % QK_K == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    return &ggml::cpu::riscv64_spacemit::q6_k_as_q4_0_32x256;
                }
                if ((ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    return &ggml::cpu::riscv64_spacemit::q6_k_32x32_q8_0;
                }
#endif
            }
            break;
        case GGML_TYPE_Q8_0:
            {
#if defined(RISCV64_SPACEMIT_IME2)
                if ((ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    return &ggml::cpu::riscv64_spacemit::q8_0_32x32_q8_0;
                }
#endif
            }
            break;
        case GGML_TYPE_MXFP4:
            {
#if defined(RISCV64_SPACEMIT_IME2)
                // TODO
                // if (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                //     return &ggml::cpu::riscv64_spacemit::mxfp4_32x32_q8_0;
                // }
#endif
            }
            break;
        case GGML_TYPE_Q5_K:
            {
#if defined(RISCV64_SPACEMIT_IME2)
                if (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    return &ggml::cpu::riscv64_spacemit::q5_k_32x32_q8_0;
                }
#endif
            }
            break;
        case GGML_TYPE_Q5_1:
            {
#if defined(RISCV64_SPACEMIT_IME2)
                if (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    return &ggml::cpu::riscv64_spacemit::q5_1_32x32_q8_0;
                }
#endif
            }
            break;
        case GGML_TYPE_Q5_0:
            {
#if defined(RISCV64_SPACEMIT_IME2)
                static const bool q5_0_ime = [] {
                    const char * e = getenv("GGML_SPACEMIT_Q5_0_IME");
                    return e && e[0] == '1';
                }();
                if (q5_0_ime && cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    return &ggml::cpu::riscv64_spacemit::q5_0_32x32_q8_0;
                }
                // default: exact conversion to Q8_0 at load (GGML_SPACEMIT_Q5_0_AS_Q8=0 keeps the Q5_0 kernel)
                static const bool q5_0_as_q8 = [] {
                    const char * e = getenv("GGML_SPACEMIT_Q5_0_AS_Q8");
                    return !(e && e[0] == '0');
                }();
                if (cur->ne[1] % 32 == 0 && cur->ne[0] % QK8_0 == 0 &&
                    (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    return q5_0_as_q8 ? (const ggml::spacemit::tensor_traits_base *) &ggml::cpu::riscv64_spacemit::q5_0_as_q8_0_32x32
                                      : &ggml::cpu::riscv64_spacemit::q5_0_32x32_q8_0;
                }
#endif
            }
            break;
        case GGML_TYPE_F32:
            {
#if defined(RISCV64_SPACEMIT_IME2)
                // init_tensor sees activations too: only weights (name "*.weight") qualify
                static const bool router_q8 = [] {
                    const char * e = getenv("GGML_SPACEMIT_ROUTER_Q8");
                    return !(e && e[0] == '0');
                }();
                const size_t nl = strlen(cur->name);
                if (router_q8 && strstr(cur->name, "ffn_gate_inp") != nullptr && strstr(cur->name, "shexp") == nullptr &&
                    nl > 7 && strcmp(cur->name + nl - 7, ".weight") == 0 && ggml_n_dims(cur) == 2 &&
                    cur->ne[1] % 32 == 0 && cur->ne[0] % QK8_0 == 0 &&
                    (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    return &ggml::cpu::riscv64_spacemit::f32_as_q8_0_32x32;
                }
#endif
            }
            break;
        default:
            break;
    }

    return nullptr;
}

size_t ggml_spacemit_nbytes(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (tensor->ne[i] <= 0) {
            return 0;
        }
    }

    GGML_UNUSED(buft);

    const auto plain_nbytes = [&]() {
        size_t total = ggml_type_size(tensor->type);
        for (int i = 0; i < GGML_MAX_DIMS; ++i) {
            total += (tensor->ne[i] - 1) * tensor->nb[i];
        }
        return total;
    };

    const size_t blck_size = ggml_blck_size(tensor->type);
    if (blck_size == 1) {
        return plain_nbytes();
    }

    const size_t row_nbytes = tensor->ne[0] * tensor->nb[0] / blck_size;

    const auto add_strided_nbytes = [&](size_t total, size_t src_block_size, size_t dst_block_size) {
        for (int i = 1; i < GGML_MAX_DIMS; ++i) {
            total += (tensor->ne[i] - 1) * (tensor->nb[i] / src_block_size) * dst_block_size;
        }
        return total;
    };

    const auto remap_block_nbytes = [&](size_t src_block_size, size_t dst_block_size, int64_t padded_rows = 0) {
        GGML_ASSERT(row_nbytes % src_block_size == 0);

        size_t total =
            add_strided_nbytes((row_nbytes / src_block_size) * dst_block_size, src_block_size, dst_block_size);

        if (padded_rows > 0 && tensor->ne[1] % padded_rows != 0) {
            total += (padded_rows - tensor->ne[1] % padded_rows) * (tensor->nb[1] / src_block_size) * dst_block_size;
        }

        return total;
    };

    size_t nbytes = row_nbytes;
    switch (tensor->type) {
        case GGML_TYPE_Q4_K:
            nbytes = remap_block_nbytes(sizeof(block_q4_K), sizeof(block_q4_1) * 8);
            break;
        case GGML_TYPE_Q6_K:
            if (ggml_spacemit_get_optimal_repack_type(tensor) == &ggml::cpu::riscv64_spacemit::q6_k_as_q4_0_32x256) {
                nbytes = remap_block_nbytes(sizeof(block_q6_K), sizeof(block_q4_0) * (QK_K / QK4_0));
            } else {
                nbytes = remap_block_nbytes(sizeof(block_q6_K), sizeof(block_q8_0) * 8, 32);
            }
            break;
        case GGML_TYPE_Q8_0:
            nbytes = remap_block_nbytes(sizeof(block_q8_0), sizeof(block_q8_0), 32);
            break;
        case GGML_TYPE_Q2_K:
            nbytes = remap_block_nbytes(sizeof(block_q2_K), sizeof(spacemit_kernels::nrow_block_q2_k<1>));
            break;
        case GGML_TYPE_Q3_K:
            nbytes = remap_block_nbytes(sizeof(block_q3_K), sizeof(spacemit_kernels::nrow_block_q3_k<1>));
            break;
        case GGML_TYPE_MXFP4:
            nbytes = remap_block_nbytes(sizeof(block_mxfp4), sizeof(spacemit_kernels::nrow_block_mxfp4<1>));
            break;
        case GGML_TYPE_Q5_K:
            nbytes = remap_block_nbytes(sizeof(block_q5_K), sizeof(spacemit_kernels::nrow_block_q5_1<1>) * 8);
            break;
        case GGML_TYPE_Q5_1:
            nbytes = remap_block_nbytes(sizeof(block_q5_1), sizeof(spacemit_kernels::nrow_block_q5_1<1>));
            break;
        case GGML_TYPE_Q5_0:
            if (ggml_spacemit_get_optimal_repack_type(tensor) == &ggml::cpu::riscv64_spacemit::q5_0_as_q8_0_32x32) {
                nbytes = remap_block_nbytes(sizeof(block_q5_0), sizeof(block_q8_0), 32);
            } else {
                nbytes = remap_block_nbytes(sizeof(block_q5_0), sizeof(spacemit_kernels::nrow_block_q5_0<1>));
            }
            break;
        default:
            nbytes = add_strided_nbytes(row_nbytes, 1, 1);
            break;
    }

    return nbytes;
}

// Graph-level fusions (concat -> ssm_conv, MoE routing weights), decided on the host before the graph launch.
void ggml_spacemit_graph_prepare(ggml_cgraph * cgraph) {
    ggml::cpu::riscv64_spacemit::graph_prepare_fusions(cgraph);
}

// RMS_NORM rms followed by MUL mul (rms, weight): true if the pair can run as one fused kernel
bool ggml_spacemit_rms_norm_mul_supported(const ggml_tensor * rms, const ggml_tensor * mul) {
    if (!ggml::cpu::riscv64_spacemit::rvv_rms_mul_enabled() || ggml::cpu::riscv64_spacemit::rvv_op_disabled(rms) ||
        mul->op != GGML_OP_MUL || (mul->src[0] != rms && mul->src[1] != rms)) {
        return false;
    }
    const ggml_tensor * w = mul->src[0] == rms ? mul->src[1] : mul->src[0];
    return rms->src[0]->type == GGML_TYPE_F32 && rms->src[0]->nb[0] == sizeof(float) && mul->type == GGML_TYPE_F32 &&
           mul->nb[0] == sizeof(float) && w->type == GGML_TYPE_F32 && w->nb[0] == sizeof(float) &&
           w->ne[0] == rms->ne[0] && ggml_are_same_shape(rms->src[0], mul);
}

void ggml_spacemit_rms_norm_mul(ggml::spacemit::context & ctx, ggml_tensor * rms, ggml_tensor * mul) {
    spacemit_kernels::rvv::forward_rms_norm_mul_f32(ctx, rms, mul);
}

const ggml::spacemit::tensor_traits_base * ggml_spacemit_get_tensor_traits(const ggml_tensor * op) {
    if (op == nullptr) {
        return nullptr;
    }

    const auto * common = &ggml::cpu::riscv64_spacemit::rvv_impl;
    switch (op->op) {
        case GGML_OP_MUL_MAT:
            if (op->src[0] && op->src[1] && ggml_n_dims(op->src[0]) == 2 &&
                op->src[1]->type == GGML_TYPE_F32) {
                const auto * traits = static_cast<const ggml::spacemit::tensor_traits_base *>(op->src[0]->extra);
                if (traits) return traits;
                const auto * repack = ggml_spacemit_get_optimal_repack_type(op->src[0]);
                if (repack) return repack;
                // F16 dense weight: same condition as compute_forward
                if (op->src[0]->type == GGML_TYPE_F16 &&
                    op->src[0]->nb[0] == sizeof(_Float16) &&
                    op->src[1]->nb[0] == sizeof(float)) {
                    return common;
                }
            }
            break;
        case GGML_OP_MUL_MAT_ID:
            if (op->src[0] && op->src[1] && ggml_n_dims(op->src[0]) == 3 &&
                op->src[1]->type == GGML_TYPE_F32) {
                const auto * traits = static_cast<const ggml::spacemit::tensor_traits_base *>(op->src[0]->extra);
                return traits ? traits : ggml_spacemit_get_optimal_repack_type(op->src[0]);
            }
            break;
        case GGML_OP_NORM:
        case GGML_OP_RMS_NORM:
            if (op->src[0] && (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16)) {
                return common;
            }
            break;
        case GGML_OP_SCALE:
            if (op->src[0] && op->src[0]->type == GGML_TYPE_F32) {
                return common;
            }
            break;
        case GGML_OP_ADD:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
            if (op->src[0] && op->src[1] &&
                (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                op->src[1]->type == op->src[0]->type && op->type == op->src[0]->type &&
                op->src[0]->nb[0] == ggml_type_size(op->src[0]->type) &&
                op->src[1]->nb[0] == ggml_type_size(op->src[1]->type) &&
                op->nb[0] == ggml_type_size(op->type) && ggml_can_repeat(op->src[1], op->src[0]) &&
                ggml_are_same_shape(op->src[0], op)) {
                return common;
            }
            break;
        case GGML_OP_FLASH_ATTN_EXT:
            if (op->src[0] && op->src[1] && op->src[2] &&
                (op->op_params[3] == GGML_PREC_F32 || op->op_params[3] == GGML_PREC_DEFAULT) &&
                (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                (op->src[1]->type == GGML_TYPE_F32 || op->src[1]->type == GGML_TYPE_F16) &&
                (op->src[2]->type == GGML_TYPE_F32 || op->src[2]->type == GGML_TYPE_F16) &&
                op->src[1]->ne[0] > 0 && op->src[2]->ne[0] > 0) {
                return common;
            }
            break;
        case GGML_OP_ROPE:
            if (op->src[0] && op->src[1]) {
                float freq_base;
                memcpy(&freq_base,  op->op_params + 5, sizeof(float));
                const int n_dims = ggml_get_op_params_i32(op, 1);
                const int mode   = ggml_get_op_params_i32(op, 2);
                const bool src2_ok = op->src[2] == nullptr ||
                    (op->src[2]->type == GGML_TYPE_F32 && ggml_is_contiguous(op->src[2]));
                const bool mode_ok = (mode == GGML_ROPE_TYPE_NORMAL ||
                                      mode == GGML_ROPE_TYPE_NEOX   ||
                                      mode == GGML_ROPE_TYPE_MROPE  ||
                                      mode == GGML_ROPE_TYPE_IMROPE);
                const bool c1 = (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16);
                const bool c2 = op->type == op->src[0]->type;
                const bool c3 = op->src[1]->type == GGML_TYPE_I32;
                const bool c4 = src2_ok;
                const bool c5 = op->src[0]->nb[0] == ggml_type_size(op->src[0]->type);
                const bool c6 = op->nb[0] == ggml_type_size(op->type);
                const bool c7 = freq_base > 0.0f;
                const bool c8 = n_dims > 0 && n_dims <= op->ne[0] && n_dims <= 512 && n_dims % 2 == 0;
                const bool c9 = mode_ok;
                if (c1 && c2 && c3 && c4 && c5 && c6 && c7 && c8 && c9) {
                    return common;
                }
            }
            break;
        case GGML_OP_CONT:
            if (op->src[0] && op->type == op->src[0]->type) {
                // permute case
                if (op->nb[0] != op->src[0]->nb[0] && op->nb[0] == op->src[0]->nb[1] &&
                    op->ne[3] * op->ne[2] * op->nb[2] == op->src[0]->ne[3] * op->src[0]->ne[2] * op->src[0]->nb[2]) {
                    return common;
                }
                // general stride-aware copy for F32/F16
                if ((op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) &&
                    ggml_is_contiguous(op) &&
                    ggml_nelements(op) == ggml_nelements(op->src[0])) {
                    return common;
                }
            }
            break;
        case GGML_OP_CPY:
            if (op->src[0] && ggml_nelements(op) == 0) {
                return common;
            }
            if (op->src[0] && op->type == op->src[0]->type && op->nb[0] == op->src[0]->nb[1] &&
                op->src[0]->nb[0] != op->src[0]->nb[1] && ggml_nelements(op->src[0]) == ggml_nelements(op)) {
                return common;
            }
            if (op->src[0] && op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32 &&
                ggml_is_contiguous(op) && ggml_nelements(op->src[0]) == ggml_nelements(op)) {
                return common;
            }
            if (op->src[0] &&
                ((op->type == GGML_TYPE_F16 && op->src[0]->type == GGML_TYPE_F32) ||
                 (op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F16)) &&
                ggml_is_contiguous(op) && ggml_is_contiguous(op->src[0]) &&
                ggml_nelements(op->src[0]) == ggml_nelements(op)) {
                return common;
            }
            break;
        case GGML_OP_SET_ROWS:
            if (op->src[0] && op->src[1] && op->src[2] &&
                (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                (op->src[1]->type == GGML_TYPE_I32 || op->src[1]->type == GGML_TYPE_I64) &&
                (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && op->src[2]->type == op->type &&
                ggml_are_same_shape(op, op->src[2]) && op->src[0]->ne[1] == op->src[1]->ne[0] &&
                op->src[0]->nb[0] == ggml_type_size(op->src[0]->type) &&
                op->src[2]->nb[0] == ggml_type_size(op->src[2]->type) && op->nb[0] == ggml_type_size(op->type) &&
                op->src[1]->nb[0] == ggml_type_size(op->src[1]->type) && op->ne[0] == op->src[0]->ne[0] &&
                op->ne[2] == op->src[0]->ne[2] && op->ne[3] == op->src[0]->ne[3] &&
                op->src[0]->ne[2] % op->src[1]->ne[1] == 0 && op->src[0]->ne[3] % op->src[1]->ne[2] == 0) {
                return common;
            }
            break;
        case GGML_OP_REPEAT:
            if (op->src[0] && (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16)) {
                const bool rows_equal = ggml_nrows(op->src[0]) == ggml_nrows(op);
                const bool n0_matches = op->src[0]->ne[0] == 1 || op->src[0]->ne[0] == op->ne[0];
                if ((rows_equal && n0_matches) ||
                    (op->src[0]->ne[1] == 1 && op->src[0]->ne[0] == op->ne[0])) {
                    return common;
                }
            }
            break;
        case GGML_OP_SUM_ROWS:
            if (op->src[0] && op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32) {
                return common;
            }
            break;
        case GGML_OP_GET_ROWS:
            if (op->src[0] && op->src[1] &&
                (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                op->type == op->src[0]->type &&
                op->src[1]->type == GGML_TYPE_I32 && op->ne[0] == op->src[0]->ne[0] &&
                op->src[0]->ne[2] == op->src[1]->ne[1] &&
                op->src[0]->nb[0] == ggml_type_size(op->src[0]->type) &&
                ggml_nrows(op) == ggml_nelements(op->src[1])) {
                return common;
            }
            break;
        case GGML_OP_CONCAT:
            if (op->src[0] && op->src[1] && ggml_get_op_params_i32(op, 0) == 0 &&
                (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                op->src[1]->type == op->src[0]->type && op->type == op->src[0]->type &&
                op->nb[0] == ggml_type_size(op->type) &&
                op->nb[1] == ggml_type_size(op->type) * (op->src[0]->ne[0] + op->src[1]->ne[0])) {
                return common;
            }
            break;
        case GGML_OP_UNARY:
            if (op->src[0] && op->src[0]->type == GGML_TYPE_F32 && ggml_is_contiguous(op->src[0]) &&
                (ggml_get_unary_op(op) == GGML_UNARY_OP_TANH    ||
                 ggml_get_unary_op(op) == GGML_UNARY_OP_GELU    ||
                 ggml_get_unary_op(op) == GGML_UNARY_OP_SILU    ||
                 ggml_get_unary_op(op) == GGML_UNARY_OP_EXP     ||
                 ggml_get_unary_op(op) == GGML_UNARY_OP_SIGMOID ||
                 ggml_get_unary_op(op) == GGML_UNARY_OP_NEG     ||
                 ggml_get_unary_op(op) == GGML_UNARY_OP_SOFTPLUS)) {
                return common;
            }
            break;
        case GGML_OP_SOFT_MAX:
            if (op->src[0] && op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 &&
                op->src[0]->nb[0] == sizeof(float) && op->nb[0] == sizeof(float)) {
                return common;
            }
            break;
        case GGML_OP_GLU:
            if (op->src[0] && op->src[0]->type == GGML_TYPE_F32) {
                if (ggml_get_glu_op(op) == GGML_GLU_OP_GEGLU) {
                    return common;
                }
                if (ggml_get_glu_op(op) == GGML_GLU_OP_SWIGLU && ggml_is_contiguous_1(op->src[0]) &&
                    op->type == GGML_TYPE_F32 && op->nb[0] == sizeof(float)) {
                    const int32_t swapped = ggml_get_op_params_i32(op, 1);
                    if (op->src[1]) {
                        if (swapped == 0 && op->src[1]->type == GGML_TYPE_F32 && ggml_is_contiguous_1(op->src[1]) &&
                            ggml_are_same_shape(op->src[0], op->src[1]) && ggml_are_same_shape(op, op->src[0])) {
                            return common;
                        }
                    } else if (op->ne[0] == op->src[0]->ne[0] / 2 && ggml_nrows(op) == ggml_nrows(op->src[0])) {
                        return common;
                    }
                }
            }
            break;
        default:
            break;
    }

    // Scalar ops: route to common so compute_forward dispatches to scalar_kernels
    switch (op->op) {
        case GGML_OP_L2_NORM:
            if (op->src[0] && op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32)
                return common;
            break;
        case GGML_OP_FILL:
            if (op->src[0] && (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16))
                return common;
            break;
        case GGML_OP_CUMSUM:
            if (op->src[0] && op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32)
                return common;
            break;
        case GGML_OP_PAD:
            if (op->src[0] && op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32)
                return common;
            break;
        case GGML_OP_TRI:
            if (op->src[0] && op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 &&
                ggml_is_contiguous(op->src[0]))
                return common;
            break;
        case GGML_OP_DIAG:
            if (op->src[0] && op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32)
                return common;
            break;
        case GGML_OP_SET:
            if (op->src[0] && op->src[1] &&
                op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
                op->type == GGML_TYPE_F32)
                return common;
            break;
        case GGML_OP_SOLVE_TRI:
            if (op->src[0] && op->src[1] &&
                op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
                op->type == GGML_TYPE_F32)
                return common;
            break;
        case GGML_OP_GATED_DELTA_NET:
            if (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2) {
                return common;
            }
            break;
        case GGML_OP_SSM_CONV:
            if (op->src[0] && op->src[0]->type == GGML_TYPE_F32)
                return common;
            break;
        case GGML_OP_ARGSORT:
            if (op->src[0] && op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_I32 &&
                op->src[0]->nb[0] == sizeof(float) && op->nb[0] == sizeof(int32_t))
                return common;
            break;
        case GGML_OP_CLAMP:
            if (op->src[0] && op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 &&
                op->src[0]->nb[0] == sizeof(float) && op->nb[0] == sizeof(float) && ggml_are_same_shape(op, op->src[0]))
                return common;
            break;
        case GGML_OP_MUL_MAT:
            // Generic CPU dot fallback for dense or unsupported quantized
            // layouts which do not have a repacked Spacemit trait.
            if (op->src[0] && op->src[1] && op->src[1]->type == GGML_TYPE_F32 &&
                ggml_get_type_traits_cpu(op->src[0]->type)->vec_dot != nullptr) {
                return common;
            }
            break;
        case GGML_OP_FLASH_ATTN_EXT:
            if (op->src[0] && op->src[1] && op->src[2] &&
                (op->op_params[3] == GGML_PREC_F32 || op->op_params[3] == GGML_PREC_DEFAULT) &&
                (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                (op->src[1]->type == GGML_TYPE_F32 || op->src[1]->type == GGML_TYPE_F16) &&
                (op->src[2]->type == GGML_TYPE_F32 || op->src[2]->type == GGML_TYPE_F16) &&
                op->src[1]->ne[0] > 0 && op->src[2]->ne[0] > 0) {
                return common;
            }
            break;
        default:
            break;
    }

    return nullptr;
}

int ggml_riscv64_spacemit_repack_tensor(ggml_tensor * tensor, const void * data, size_t size) {
    auto traits = (ggml::spacemit::tensor_traits_base *) tensor->extra;
    if (traits) {
        return traits->repack(tensor, data, size);
    }
    memcpy(tensor->data, data, size);
    return 0;
}

bool ggml_spacemit_get_work_size(int n_threads, const ggml_tensor * op, size_t * size) {
    const auto * traits = ggml_spacemit_get_tensor_traits(op);
    return traits != nullptr && traits->work_size(n_threads, op, *size);
}

bool ggml_spacemit_compute_forward(ggml::spacemit::context & ctx, ggml_tensor * op) {
    const auto * traits = ggml_spacemit_get_tensor_traits(op);
    return traits != nullptr && traits->compute_forward(ctx, op);
}
