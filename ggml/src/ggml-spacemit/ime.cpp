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
#include "spacemit-fused.h"
#include "spacemit-opnode.h"
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
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
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

template <typename BLOC_TYPE, int64_t INTER_SIZE, int64_t NB_COLS>
class tensor_traits : public ggml::spacemit::tensor_traits_base {
    bool work_size(int /* n_threads */, const ggml_tensor * op, size_t & size) const override {
        switch (op->op) {
            case GGML_OP_MUL_MAT:
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
                        return forward_mul_mat(ctx, op);
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
                        return forward_mul_mat_id(ctx, op);
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

    bool forward_mul_mat(ggml::spacemit::context & ctx, ggml_tensor * op) const {
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

        int64_t block_stride_a = spacemit_kernels::q8_blk_size(a_blk_len);

#if defined(RISCV64_SPACEMIT_IME2)
        if (!set_kernel_impl && (global_spine_env_info.use_ime2)) {
            quantize_a_row_i8  = spacemit_kernels::rvv::quantize_a_row_i8;
            quantize_a_4row_i8 = spacemit_kernels::rvv::quantize_a_4row_i8;
            block_stride_a     = spacemit_kernels::q8_blk_size(a_blk_len, true);

            if constexpr (std::is_same_v<BLOC_TYPE, block_q6_K> || std::is_same_v<BLOC_TYPE, block_q8_0>) {
                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8i8;
                set_kernel_impl = true;
            } else if constexpr (std::is_same_v<BLOC_TYPE, block_q4_0> || std::is_same_v<BLOC_TYPE, block_q4_1> ||
                                 std::is_same_v<BLOC_TYPE, block_q4_K>) {
                if constexpr (INTER_SIZE == 256) {
                    gemm_kernel        = spacemit_kernels::ime2::gemm_kernel_i8i4_hp;
                    quantize_a_row_i8  = spacemit_kernels::rvv::quantize_a_row_i8_hp;
                    quantize_a_4row_i8 = spacemit_kernels::rvv::quantize_a_4row_i8_hp;
                    block_stride_a     = spacemit_kernels::q8_hp_blk_size(a_blk_len, true, true);
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
        const int64_t gemm_workspace_size = GGML_PAD(gemm_m * row_stride_a, alignof(int64_t));

        if (ith == 0 && ctx.workspace_size < gemm_workspace_size) {
            GGML_ABORT("wsize less than gemm_workspace_size");
        }

        uintptr_t ws_ptr = reinterpret_cast<uintptr_t>(ctx.workspace);

        void *        tcm_buffer      = ctx.shared.data;
        const int64_t tcm_buffer_size = ctx.shared.size;

        auto * quant_a_buffer = reinterpret_cast<uint8_t *>(ws_ptr);

        constexpr int64_t row_align = 4;
        const int64_t     row_blks  = spacemit_kernels::div_round_up(gemm_m, row_align);

        const int64_t row_stride_b      = b_k_blks * get_repacked_block_type_size<BLOC_TYPE, INTER_SIZE, NB_COLS>();
        const int64_t per_mb_rows_wsize = row_align * row_stride_a;
        const int64_t per_nb_cols_wsize = NB_COLS * row_stride_b;

        const int64_t barrier_idx = static_cast<int64_t>(ith / 2);

        GGML_ASSERT(global_spine_env_info.init_barrier != nullptr);
        GGML_ASSERT(barrier_idx < spine_init_barrier_count);
        spine_barrier_t * cur_barrier = &global_spine_env_info.init_barrier[barrier_idx];

        if (gemm_m == 1) {
            int task_per_thread = spacemit_kernels::div_round_up(a_k_blks, nth);
            int a_blk_start     = ith * task_per_thread;
            int a_blk_end       = std::min(a_blk_start + task_per_thread, (int) a_k_blks);
            if (a_blk_start < a_blk_end) {
                quantize_a_row_i8(a_blk_len, feature + a_blk_start * a_blk_len, (a_blk_end - a_blk_start) * a_blk_len,
                                  quant_a_buffer + a_blk_start * block_stride_a);
            }
        } else {
            int task_per_thread = spacemit_kernels::div_round_up(row_blks, nth);
            int m_row_blk_start = ith * task_per_thread;
            int m_row_blk_end   = std::min(m_row_blk_start + task_per_thread, (int) row_blks);
            for (int m_row_blk = m_row_blk_start; m_row_blk < m_row_blk_end; m_row_blk++) {
                int m_idx             = m_row_blk * row_align;
                int rows_tobe_handled = (gemm_m - m_idx) > row_align ? row_align : (gemm_m - m_idx);

                if (rows_tobe_handled == row_align && quantize_a_4row_i8 != nullptr) {
                    const float * a_row_ptr       = feature + m_idx * gemm_k;
                    auto *        quant_a_row_ptr = quant_a_buffer + m_idx * row_stride_a;
                    quantize_a_4row_i8(a_blk_len, a_row_ptr, gemm_k, quant_a_row_ptr);
                } else {
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

        ctx.sync();

        const int64_t gemm_m_stride     = gemm_n / gemm_m > 64 ? gemm_m : 16;
        const int64_t gemm_m_blocked    = spacemit_kernels::div_round_up(gemm_m, gemm_m_stride);
        const int64_t max_gemm_n_stride = spacemit_kernels::div_round_up(gemm_n * gemm_m_blocked, nth);

        int64_t gemm_n_stride = gemm_n;
        if (max_gemm_n_stride < gemm_n) {
            gemm_n_stride =
                std::min(gemm_n_stride, spacemit_kernels::div_round_up(max_gemm_n_stride, NB_COLS) * NB_COLS);
        }

        if (gemm_n_stride == gemm_n && tcm_buffer != nullptr && per_mb_rows_wsize <= tcm_buffer_size) {
            for (int64_t m_start = ith * row_align; m_start < gemm_m; m_start += row_align * nth) {
                uint8_t * b_col    = reinterpret_cast<uint8_t *>(w_data);
                uint8_t * b_col_zp = block_type_has_zp<BLOC_TYPE>() ? b_col : nullptr;

                int64_t m_row_real = std::min(gemm_m - m_start, row_align);

                spacemit_kernels::rvv::memcpy1d(tcm_buffer, quant_a_buffer + m_start * row_stride_a,
                                                m_row_real * row_stride_a);

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
            }
        } else if (tcm_buffer != nullptr && per_nb_cols_wsize <= tcm_buffer_size) {
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
                    return true;
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
        return true;
    }

    bool forward_mul_mat_id(ggml::spacemit::context & ctx, ggml_tensor * op) const {
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

        struct mmid_row_mapping {
            int32_t i1;
            int32_t i2;
        };

        spacemit_kernels::quantize_a_row_def           quantize_a_row_i8;
        spacemit_kernels::gemm_kernel_quantize_def     gemm_kernel;
        spacemit_kernels::moe_gemm_kernel_quantize_def moe_gemm_kernel_m2;
        bool                                           set_kernel_impl = false;
        size_t                                         block_stride_a  = spacemit_kernels::q8_blk_size(QK4_0);

#if defined(RISCV64_SPACEMIT_IME2)
        if (!set_kernel_impl && (global_spine_env_info.use_ime2)) {
            quantize_a_row_i8 = spacemit_kernels::rvv::quantize_a_row_i8;
            block_stride_a    = spacemit_kernels::q8_blk_size(QK4_0, true);

            if constexpr (std::is_same_v<BLOC_TYPE, block_q6_K> || std::is_same_v<BLOC_TYPE, block_q8_0>) {
                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8i8;
                set_kernel_impl = true;
            } else if constexpr (std::is_same_v<BLOC_TYPE, block_q4_0> || std::is_same_v<BLOC_TYPE, block_q4_1> ||
                                 std::is_same_v<BLOC_TYPE, block_q4_K>) {
                if constexpr (INTER_SIZE == 256) {
                    gemm_kernel       = spacemit_kernels::ime2::gemm_kernel_i8i4_hp;
                    quantize_a_row_i8 = spacemit_kernels::rvv::quantize_a_row_i8_hp;
                    block_stride_a    = spacemit_kernels::q8_hp_blk_size(a_blk_len, true, true);
                    set_kernel_impl   = true;
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

        if (ne11 == 1) {
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

        int64_t *          matrix_row_counts       = (int64_t *) (ws_ptr + gemm_workspace_size);
        int32_t *          valid_ep_count          = (int32_t *) (matrix_row_counts + n_as);
        int32_t *          valid_act_count         = (int32_t *) (valid_ep_count + 1);
        int64_t *          valid_matrix_row_counts = (int64_t *) (valid_act_count + 1);
        mmid_row_mapping * matrix_rows             = (mmid_row_mapping *) (valid_matrix_row_counts + n_as);

        if (ith == 0) {
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
        }

        ctx.sync();

        const size_t row_stride_b      = b_k_blks * get_repacked_block_type_size<BLOC_TYPE, INTER_SIZE, NB_COLS>();
        const size_t expert_b_stride   = ne01 * row_stride_b;
        const size_t per_nb_cols_wsize = NB_COLS * row_stride_b;

        const int64_t barrier_idx = static_cast<int64_t>(ith / 2);

        GGML_ASSERT(global_spine_env_info.init_barrier != nullptr);
        GGML_ASSERT(barrier_idx < spine_init_barrier_count);
        spine_barrier_t * cur_barrier = &global_spine_env_info.init_barrier[barrier_idx];

        std::array<const uint8_t *, 2> src_workspaces;
        std::array<float *, 2>         dst_workspaces;

        auto *     tcm_buffer      = ctx.shared.data;
        const auto tcm_buffer_size = ctx.shared.size;

        const auto valid_ep_count_t  = valid_ep_count[0];
        const auto valid_act_count_t = valid_act_count[0];

        int nth_es = 1;
        int nth_n  = nth;

        int ith_es = ith % nth_es;
        int ith_n  = (ith / nth_es) % nth_n;

        if (valid_ep_count_t % nth == 0 && tcm_buffer != nullptr && valid_ep_count_t == n_as &&
            valid_act_count_t == n_as && per_nb_cols_wsize <= tcm_buffer_size) {
            const bool has_pair = ((ith & 1) != 0) || (ith + 1 < nth);

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
                if (tcm_buffer != nullptr && (src1_cur_end - src1_cur_start) >= 4 &&
                    (src0_cur_end - src0_cur_start) * row_stride_b <= tcm_buffer_size) {
                    spacemit_kernels::rvv::memcpy1d(tcm_buffer, src0_cur,
                                                    (src0_cur_end - src0_cur_start) * row_stride_b);
                    src0_cur = reinterpret_cast<uint8_t *>(tcm_buffer);
                    b_col_zp = block_type_has_zp<BLOC_TYPE>() ? src0_cur : nullptr;
                    extra_tcm_buffer_size -= (src0_cur_end - src0_cur_start) * row_stride_b;
                    extra_tcm_buffer = reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(tcm_buffer) +
                                                                (src0_cur_end - src0_cur_start) * row_stride_b);
                }

                int ir1 = src1_cur_start;

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
                            }
                        }

                        for (; iir1 < (ir1 + quant_a_tile_size); iir1++, quant_a_tile_buffer += nbw1) {
                            mmid_row_mapping row_mapping_0 = MMID_MATRIX_ROW(cur_a, iir1);

                            gemm_kernel(
                                b_blk_len, quant_a_tile_buffer, src0_cur, b_col_zp,
                                (float *) ((char *) dst->data + (row_mapping_0.i1 * nb1 + row_mapping_0.i2 * nb2)) +
                                    src0_cur_start,
                                1, src0_cur_end - src0_cur_start, b_k_blks, ne01);
                        }

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
                        }
                    }

                    for (; ir1 < src1_cur_end; ir1++) {
                        mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, ir1);

                        const int id = row_mapping.i1;  // selected expert index

                        const int64_t i11 = id % ne11;
                        const int64_t i12 = row_mapping.i2;  // row index in src1

                        const int64_t i1 = id;               // selected expert index
                        const int64_t i2 = i12;              // row

                        auto * src1_col = quant_a_buffer + (i11 * nbw1 + i12 * nbw2);

                        gemm_kernel(b_blk_len, src1_col, src0_cur, b_col_zp,
                                    (float *) ((char *) dst->data + (i1 * nb1 + i2 * nb2)) + src0_cur_start, 1,
                                    src0_cur_end - src0_cur_start, b_k_blks, ne01);
                    }
                }
            }
        }
#undef MMID_MATRIX_ROW
        return true;
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
                    // row-contiguous F32 copy with arbitrary outer strides (GDN rollback snapshots)
                    if (op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32 &&
                        op->nb[0] == sizeof(float) && src0->nb[0] == sizeof(float) &&
                        op->ne[0] == src0->ne[0] && op->ne[1] == src0->ne[1] &&
                        op->ne[2] == src0->ne[2] && op->ne[3] == src0->ne[3]) {
                        spacemit_kernels::rvv::forward_cpy_f32_row_strided(ctx, op);
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
                    if (spacemit_kernels::rvv::forward_concat_dim0_transposed_f32(ctx, op, 0)) {
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
                if (!spacemit_kernels::rvv::forward_ssm_conv_f32(ctx, op)) {
                    spacemit_kernels::scalar::forward_ssm_conv_f32(ctx, op);
                }
                return true;
            default:
                break;
        }
        return false;
    }

    bool compute_forward_fused(ggml::spacemit::context & ctx, const spacemit_opnode & node) const override {
        switch (node.opcode) {
            case SPACEMIT_OP_RMS_NORM_MUL:
                // out = rms_norm(x) * w in one pass; node.node = RMS_NORM, node.fused[0] = MUL
                spacemit_kernels::rvv::forward_rms_norm_mul_f32(ctx, node.node, node.fused[0]);
                return true;
            default:
                return false;
        }
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
        // read once, then merge the partials (ported from ggml-cpu/spacemit)
        const int n_split = (!use_tiled && ctx.shared.data != nullptr) ?
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

#if defined(RISCV64_SPACEMIT_IME2)
// Per-row INT8 GEMM for Q8_0 weights whose 32-blocks share one scale per row.
//
// Such a tensor is per-output-channel INT8 stored in the Q8_0 container. The
// int32 accumulator can then run through the whole of k: a 16x32 output tile
// lives in 16 accumulator registers with no float conversion in the k loop.
// The GEMM is split by output columns, so every worker streams only its own
// weight columns from DRAM, once, and reuses them across all M tiles from TCM.
//
// Repacked weight layout (t->data), N32 = ceil(N / 32):
//   strips[N32][K/16][32 cols][16 B]  int8; one 512 B block per 16 k is four
//                                      vregs of 8 columns x 16 k each
//   scales[N32 * 32]                  f32, one per output row of the weight
// A workspace, M16 = ceil(M / 16) * 16:
//   groups[M16/16][K/16][16 rows][16 B] int8, then scales[M16] f32
namespace perrow_q8 {

// GGML_SPACEMIT_PERROW: 0 off; 1 (default) Q8_0 weights whose 32-blocks share one
// scale per row; 2 every Q8_0 or F16 2D weight under blk.* (requantized per row).
static int mode() {
    static const int m = [] {
        const char * env = std::getenv("GGML_SPACEMIT_PERROW");
        return env ? std::atoi(env) : 1;
    }();
    return m;
}

static bool enabled() {
    return mode() != 0;
}

// GGML_SPACEMIT_PERROW_ROT: orthogonal rotation along k applied to both weights
// (at load) and activations (before quantization), so outlier channels are
// spread over the whole row and one INT8 scale per row fits. 0 none; 1 Hadamard
// across 32-wide rows only (H_R x I_32); 2 (default) full Walsh-Hadamard over
// blocks of B = the largest power of two dividing K, at most 2048.
static int rot_mode() {
    static const int m = [] {
        const char * env = std::getenv("GGML_SPACEMIT_PERROW_ROT");
        return env ? std::atoi(env) : 2;
    }();
    return m;
}

static int64_t rot_block(int64_t K) {
    int64_t b = 1;
    while (K % (b * 2) == 0 && b * 2 <= 2048) {
        b *= 2;
    }
    return rot_mode() != 0 && b >= 64 ? b : 0;
}

// Butterflies with distance >= this are applied; 1 for the full transform.
static int64_t rot_h0() {
    return rot_mode() == 1 ? 32 : 1;
}

// Scalar reference rotation, orthonormal. Used for the weights at load time.
static void rotate_ref(float * x, int64_t K) {
    const int64_t B = rot_block(K);
    if (B == 0) {
        return;
    }
    const int64_t h0 = rot_h0();
    const float   sc = 1.0f / std::sqrt((float) (B / h0));
    for (int64_t b = 0; b < K; b += B) {
        float * v = x + b;
        for (int64_t h = h0; h < B; h *= 2) {
            for (int64_t i = 0; i < B; i += 2 * h) {
                for (int64_t j = i; j < i + h; ++j) {
                    const float a = v[j], c = v[j + h];
                    v[j]          = a + c;
                    v[j + h]      = a - c;
                }
            }
        }
        for (int64_t j = 0; j < B; ++j) {
            v[j] *= sc;
        }
    }
}

static float half_to_float(ggml_half h) {
    _Float16 f;
    memcpy(&f, &h, sizeof(f));
    return (float) f;
}

static bool weight_name(const ggml_tensor * t) {
    return strncmp(t->name, "blk.", 4) == 0;
}

static bool shape_ok(const ggml_tensor * t) {
    return t->ne[2] == 1 && t->ne[3] == 1 && t->ne[0] >= 64 && t->ne[0] % 64 == 0;
}

static bool eligible(const ggml_tensor * t, const void * data, size_t data_size) {
    if (!enabled() || t->type != GGML_TYPE_Q8_0 || !shape_ok(t)) {
        return false;
    }
    const int64_t nblk = t->ne[0] / QK8_0;
    const int64_t nrow = t->ne[1];
    if (data_size != (size_t) (nrow * nblk) * sizeof(block_q8_0)) {
        return false;
    }
    if (mode() == 2) {
        return weight_name(t);
    }
    const auto * src = static_cast<const block_q8_0 *>(data);
    for (int64_t r = 0; r < nrow; ++r) {
        const block_q8_0 * row = src + r * nblk;
        for (int64_t b = 1; b < nblk; ++b) {
            if (row[b].d != row[0].d) {
                return false;
            }
        }
    }
    return true;
}

// F16 weights are taken on metadata alone (the traits are chosen before the data arrives).
static bool eligible_f16(const ggml_tensor * t) {
    return mode() == 2 && t->type == GGML_TYPE_F16 && shape_ok(t) && weight_name(t) && ggml_n_dims(t) == 2;
}

// Weights as N32 strips of [K/16][32 cols][16 B] followed by one f32 scale per row.
// row(n, out) writes row n of the source as K floats. Rows are rotated, then
// quantized to symmetric INT8 with one scale each. A uniform-scale Q8_0 source
// with rotation off is copied bit-exactly instead.
template <typename Row>
static void repack_rows(ggml_tensor * t, const Row & row) {
    const int64_t K      = t->ne[0];
    const int64_t N      = t->ne[1];
    const int64_t nstrip = (N + 31) / 32;
    auto *        dst    = static_cast<int8_t *>(t->data);
    std::vector<float> scales(nstrip * 32, 0.0f);
    // Each strip is assembled in host memory and written to the tensor buffer in
    // one sequential copy: scattered 16 B stores straight into the backend buffer
    // ran at ~10 MB/s on K3. Strips are split across threads (the load runs on
    // the X100 cores; the rotation is ~11 butterfly passes per weight).
    const int nth = (int) std::max(1u, std::min(8u, std::thread::hardware_concurrency()));
    auto      work = [&](int ith) {
        std::vector<int8_t> strip(32 * K);
        std::vector<float>  f(K);
        std::vector<int8_t> q(K);
        for (int64_t s = ith; s < nstrip; s += nth) {
            for (int64_t c = 0; c < 32; ++c) {
                const int64_t n = s * 32 + c;
                if (n >= N) {
                    for (int64_t kb = 0; kb < K / 16; ++kb) {
                        memset(strip.data() + kb * 512 + c * 16, 0, 16);
                    }
                    continue;
                }
                scales[n] = row(n, f.data(), q.data());
                for (int64_t kb = 0; kb < K / 16; ++kb) {
                    memcpy(strip.data() + kb * 512 + c * 16, q.data() + kb * 16, 16);
                }
            }
            memcpy(dst + s * 32 * K, strip.data(), strip.size());
        }
    };
    std::vector<std::thread> pool;
    for (int i = 1; i < nth; ++i) {
        pool.emplace_back(work, i);
    }
    work(0);
    for (auto & th : pool) {
        th.join();
    }
    memcpy(dst + nstrip * 32 * K, scales.data(), scales.size() * sizeof(float));
}

// Rotate f (K floats) and quantize it per row into q; returns the scale.
static float rotate_quantize(float * f, int64_t K, int8_t * q) {
    rotate_ref(f, K);
    float amax = 0.0f;
    for (int64_t k = 0; k < K; ++k) {
        amax = std::max(amax, std::fabs(f[k]));
    }
    const float d  = amax / 127.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    for (int64_t k = 0; k < K; ++k) {
        q[k] = (int8_t) std::max(-127.0f, std::min(127.0f, std::nearbyint(f[k] * id)));
    }
    return d;
}

static void repack(ggml_tensor * t, const block_q8_0 * src) {
    const int64_t K    = t->ne[0];
    const int64_t nblk = K / QK8_0;
    const bool    copy = rot_block(K) == 0 && mode() == 1;
    repack_rows(t, [&](int64_t n, float * f, int8_t * q) {
        const block_q8_0 * row = src + n * nblk;
        if (copy) {
            for (int64_t b = 0; b < nblk; ++b) {
                memcpy(q + b * QK8_0, row[b].qs, QK8_0);
            }
            return half_to_float(row[0].d);
        }
        for (int64_t b = 0; b < nblk; ++b) {
            const float d = half_to_float(row[b].d);
            for (int j = 0; j < QK8_0; ++j) {
                f[b * QK8_0 + j] = d * row[b].qs[j];
            }
        }
        return rotate_quantize(f, K, q);
    });
}

static void repack_f16(ggml_tensor * t, const ggml_half * src) {
    const int64_t K = t->ne[0];
    repack_rows(t, [&](int64_t n, float * f, int8_t * q) {
        for (int64_t k = 0; k < K; ++k) {
            f[k] = half_to_float(src[n * K + k]);
        }
        return rotate_quantize(f, K, q);
    });
}

// 16x32 int32 tile over `blocks` 16-k blocks: a is 16 rows x 16 B per block,
// b is 32 columns x 16 B per block; tmp receives 16 rows x 32 int32, row-major.
// Two A and two B register sets are software-pipelined so each load overlaps
// the eight vmadot of the previous block.
__attribute__((noinline)) static void tile_m16n32(const int8_t * a, const int8_t * b, int64_t blocks,
                                                  int32_t * tmp) {
    asm volatile(
        "vsetvli t0, zero, e32, m8, ta, ma\n\t"
        "vmv.v.i v16, 0\n\t"
        "vmv.v.i v24, 0\n\t"
        "vsetvli t0, zero, e8, m1, ta, ma\n\t"
        "vl2re8.v v0, (%[a])\n\t"
        "vl4re8.v v4, (%[b])\n\t"
        "addi %[a], %[a], 256\n\t"
        "addi %[b], %[b], 512\n\t"
        "1:\n\t"
        "li t0, 1\n\t"
        "beq %[blocks], t0, 3f\n\t"
        "vl2re8.v v2, (%[a])\n\t"
        "addi %[a], %[a], 256\n\t"
        "vmadot v16, v0, v4, i8\n\t"
        "vmadot v24, v1, v4, i8\n\t"
        "vl4re8.v v8, (%[b])\n\t"
        "addi %[b], %[b], 512\n\t"
        "vmadot v18, v0, v5, i8\n\t"
        "vmadot v26, v1, v5, i8\n\t"
        "vmadot v20, v0, v6, i8\n\t"
        "vmadot v28, v1, v6, i8\n\t"
        "vmadot v22, v0, v7, i8\n\t"
        "vmadot v30, v1, v7, i8\n\t"
        "addi %[blocks], %[blocks], -1\n\t"
        "beq %[blocks], t0, 2f\n\t"
        "vl2re8.v v0, (%[a])\n\t"
        "addi %[a], %[a], 256\n\t"
        "vmadot v16, v2, v8, i8\n\t"
        "vmadot v24, v3, v8, i8\n\t"
        "vl4re8.v v4, (%[b])\n\t"
        "addi %[b], %[b], 512\n\t"
        "vmadot v18, v2, v9, i8\n\t"
        "vmadot v26, v3, v9, i8\n\t"
        "vmadot v20, v2, v10, i8\n\t"
        "vmadot v28, v3, v10, i8\n\t"
        "vmadot v22, v2, v11, i8\n\t"
        "vmadot v30, v3, v11, i8\n\t"
        "addi %[blocks], %[blocks], -1\n\t"
        "j 1b\n\t"
        "2:\n\t"
        "vmadot v16, v2, v8, i8\n\t"
        "vmadot v24, v3, v8, i8\n\t"
        "vmadot v18, v2, v9, i8\n\t"
        "vmadot v26, v3, v9, i8\n\t"
        "vmadot v20, v2, v10, i8\n\t"
        "vmadot v28, v3, v10, i8\n\t"
        "vmadot v22, v2, v11, i8\n\t"
        "vmadot v30, v3, v11, i8\n\t"
        "j 4f\n\t"
        "3:\n\t"
        "vmadot v16, v0, v4, i8\n\t"
        "vmadot v24, v1, v4, i8\n\t"
        "vmadot v18, v0, v5, i8\n\t"
        "vmadot v26, v1, v5, i8\n\t"
        "vmadot v20, v0, v6, i8\n\t"
        "vmadot v28, v1, v6, i8\n\t"
        "vmadot v22, v0, v7, i8\n\t"
        "vmadot v30, v1, v7, i8\n\t"
        "4:\n\t"
        "vsetvli t0, zero, e32, m1, ta, ma\n\t"
        "vpack.vv v0, v16, v18, 2\n\t"
        "vpack.vv v2, v20, v22, 2\n\t"
        "vpack.vv v4, v17, v19, 2\n\t"
        "vpack.vv v6, v21, v23, 2\n\t"
        "vpack.vv v8, v24, v26, 2\n\t"
        "vpack.vv v10, v28, v30, 2\n\t"
        "vpack.vv v12, v25, v27, 2\n\t"
        "vpack.vv v14, v29, v31, 2\n\t"
        "vpack.vv v16, v0, v2, 3\n\t"
        "vpack.vv v18, v1, v3, 3\n\t"
        "vpack.vv v20, v4, v6, 3\n\t"
        "vpack.vv v22, v5, v7, 3\n\t"
        "vpack.vv v24, v8, v10, 3\n\t"
        "vpack.vv v26, v9, v11, 3\n\t"
        "vpack.vv v28, v12, v14, 3\n\t"
        "vpack.vv v30, v13, v15, 3\n\t"
        "vs8r.v v16, (%[tmp])\n\t"
        "addi %[tmp], %[tmp], 1024\n\t"
        "vs8r.v v24, (%[tmp])\n\t"
        : [a] "+&r"(a), [b] "+&r"(b), [blocks] "+&r"(blocks), [tmp] "+&r"(tmp)
        :
        : "t0", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13",
          "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28",
          "v29", "v30", "v31");
}

// rotate_ref without the normalization (folded into the row scale), on the A100
// (VLEN 1024). Distances >= 32 are butterflies between contiguous runs;
// distances 1..16 stay inside 128-element register groups and use slides.
static void rotate_vec_generic(float * v, int64_t K) {
    const int64_t B  = rot_block(K);
    const int64_t h0 = rot_h0();
    for (int64_t b0 = 0; b0 < K; b0 += B) {
        float * x = v + b0;
        for (int64_t h = std::max<int64_t>(h0, 32); h < B; h *= 2) {
            for (int64_t i = 0; i < B; i += 2 * h) {
                for (int64_t j = 0; j < h;) {
                    const size_t vl = __riscv_vsetvl_e32m8(h - j);
                    const auto   a  = __riscv_vle32_v_f32m8(x + i + j, vl);
                    const auto   c  = __riscv_vle32_v_f32m8(x + i + j + h, vl);
                    __riscv_vse32_v_f32m8(x + i + j, __riscv_vfadd_vv_f32m8(a, c, vl), vl);
                    __riscv_vse32_v_f32m8(x + i + j + h, __riscv_vfsub_vv_f32m8(a, c, vl), vl);
                    j += vl;
                }
            }
        }
        if (h0 < 32) {
            const size_t vl = __riscv_vsetvl_e32m4((size_t) std::min<int64_t>(B, 128));
            const auto   id = __riscv_vid_v_u32m4(vl);
            const auto   hi = [&](unsigned h) { return __riscv_vmsne_vx_u32m4_b8(__riscv_vand_vx_u32m4(id, h, vl), 0, vl); };
            const auto   m1 = hi(1), m2 = hi(2), m4 = hi(4), m8 = hi(8), m16 = hi(16);
            const auto   stage = [&](vfloat32m4_t xv, size_t h, vbool8_t m) {
                const auto dn = __riscv_vslidedown_vx_f32m4(xv, h, vl);    // x[i + h]
                const auto up = __riscv_vslideup_vx_f32m4(xv, xv, h, vl);  // x[i - h]
                return __riscv_vmerge_vvm_f32m4(__riscv_vfadd_vv_f32m4(xv, dn, vl), __riscv_vfsub_vv_f32m4(up, xv, vl),
                                                m, vl);
            };
            for (int64_t c0 = 0; c0 < B; c0 += vl) {
                auto xv = __riscv_vle32_v_f32m4(x + c0, vl);
                xv      = stage(xv, 1, m1);
                xv      = stage(xv, 2, m2);
                xv      = stage(xv, 4, m4);
                xv      = stage(xv, 8, m8);
                xv      = stage(xv, 16, m16);
                __riscv_vse32_v_f32m4(x + c0, xv, vl);
            }
        }
    }
}

// Fast path for B >= 256 on VLEN 1024 (256 floats per e32m8 group): each
// 256-float chunk is read from src once and gets every butterfly with distance
// < 256 in registers (1..16 by slides, 32/64/128 between its eight vregs); the
// distances 256..B/2 then run on x in memory. src and x may be the same.
static void rotate_vec(const float * src, float * x, int64_t K) {
    const int64_t B = rot_block(K);
    if (B < 256 || __riscv_vsetvlmax_e32m8() != 256) {
        if (src != x) {
            memcpy(x, src, K * sizeof(float));
        }
        rotate_vec_generic(x, K);
        return;
    }
    const bool   full = rot_h0() == 1;
    const size_t vl   = 256;
    const auto   id   = __riscv_vid_v_u32m8(vl);
    const auto   hi   = [&](unsigned h) { return __riscv_vmsne_vx_u32m8_b4(__riscv_vand_vx_u32m8(id, h, vl), 0, vl); };
    const auto   m1 = hi(1), m2 = hi(2), m4 = hi(4), m8 = hi(8), m16 = hi(16);
    const auto   stage = [&](vfloat32m8_t xv, size_t h, vbool4_t m) {
        const auto dn = __riscv_vslidedown_vx_f32m8(xv, h, vl);    // x[i + h]
        const auto up = __riscv_vslideup_vx_f32m8(xv, xv, h, vl);  // x[i - h]
        return __riscv_vmerge_vvm_f32m8(__riscv_vfadd_vv_f32m8(xv, dn, vl), __riscv_vfsub_vv_f32m8(up, xv, vl), m,
                                        vl);
    };
    const size_t v1 = 32;
    const auto   bf = [&](vfloat32m1_t & a, vfloat32m1_t & c) {
        const auto t = __riscv_vfadd_vv_f32m1(a, c, v1);
        c            = __riscv_vfsub_vv_f32m1(a, c, v1);
        a            = t;
    };
    for (int64_t c0 = 0; c0 < K; c0 += 256) {
        auto xv = __riscv_vle32_v_f32m8(src + c0, vl);
        if (full) {
            xv = stage(xv, 1, m1);
            xv = stage(xv, 2, m2);
            xv = stage(xv, 4, m4);
            xv = stage(xv, 8, m8);
            xv = stage(xv, 16, m16);
        }
        auto r0 = __riscv_vget_v_f32m8_f32m1(xv, 0), r1 = __riscv_vget_v_f32m8_f32m1(xv, 1);
        auto r2 = __riscv_vget_v_f32m8_f32m1(xv, 2), r3 = __riscv_vget_v_f32m8_f32m1(xv, 3);
        auto r4 = __riscv_vget_v_f32m8_f32m1(xv, 4), r5 = __riscv_vget_v_f32m8_f32m1(xv, 5);
        auto r6 = __riscv_vget_v_f32m8_f32m1(xv, 6), r7 = __riscv_vget_v_f32m8_f32m1(xv, 7);
        bf(r0, r1), bf(r2, r3), bf(r4, r5), bf(r6, r7);
        bf(r0, r2), bf(r1, r3), bf(r4, r6), bf(r5, r7);
        bf(r0, r4), bf(r1, r5), bf(r2, r6), bf(r3, r7);
        xv = __riscv_vset_v_f32m1_f32m8(xv, 0, r0);
        xv = __riscv_vset_v_f32m1_f32m8(xv, 1, r1);
        xv = __riscv_vset_v_f32m1_f32m8(xv, 2, r2);
        xv = __riscv_vset_v_f32m1_f32m8(xv, 3, r3);
        xv = __riscv_vset_v_f32m1_f32m8(xv, 4, r4);
        xv = __riscv_vset_v_f32m1_f32m8(xv, 5, r5);
        xv = __riscv_vset_v_f32m1_f32m8(xv, 6, r6);
        xv = __riscv_vset_v_f32m1_f32m8(xv, 7, r7);
        __riscv_vse32_v_f32m8(x + c0, xv, vl);
    }
    for (int64_t b0 = 0; b0 < K; b0 += B) {
        float * v = x + b0;
        for (int64_t h = 256; h < B; h *= 2) {
            for (int64_t i = 0; i < B; i += 2 * h) {
                for (int64_t j = 0; j < h; j += 256) {
                    const auto a = __riscv_vle32_v_f32m8(v + i + j, vl);
                    const auto c = __riscv_vle32_v_f32m8(v + i + j + h, vl);
                    __riscv_vse32_v_f32m8(v + i + j, __riscv_vfadd_vv_f32m8(a, c, vl), vl);
                    __riscv_vse32_v_f32m8(v + i + j + h, __riscv_vfsub_vv_f32m8(a, c, vl), vl);
                }
            }
        }
    }
}

// Quantize one activation row to symmetric INT8 with a single scale and scatter
// its 16 B k-chunks into row `rr` of a 16-row A group. `scratch` is 5K bytes:
// the rotated row as floats, then the K int8 codes.
static void quantize_row(const float * x, int64_t K, int8_t * group, int64_t rr, float * scale, uint8_t * scratch) {
    auto *      f    = reinterpret_cast<float *>(scratch);
    auto *      line = reinterpret_cast<int8_t *>(scratch + K * sizeof(float));
    const int64_t B  = rot_block(K);
    float       norm = 1.0f;
    if (B != 0) {
        rotate_vec(x, f, K);
        x    = f;
        norm = 1.0f / std::sqrt((float) (B / rot_h0()));
    }
    const size_t vlmax = __riscv_vsetvlmax_e32m8();
    vfloat32m8_t mx    = __riscv_vfmv_v_f_f32m8(0.0f, vlmax);
    for (int64_t c = 0; c < K;) {
        const size_t vl = __riscv_vsetvl_e32m8(K - c);
        mx = __riscv_vfmax_vv_f32m8_tu(mx, mx, __riscv_vfabs_v_f32m8(__riscv_vle32_v_f32m8(x + c, vl), vl), vl);
        c += vl;
    }
    const float amax =
        __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmax_vs_f32m8_f32m1(mx, __riscv_vfmv_v_f_f32m1(0.0f, 1), vlmax));
    const float d  = amax / 127.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    *scale         = d * norm;
    for (int64_t c = 0; c < K;) {
        const size_t vl = __riscv_vsetvl_e32m8(K - c);
        auto         q  = __riscv_vfcvt_x_f_v_i32m8(__riscv_vfmul_vf_f32m8(__riscv_vle32_v_f32m8(x + c, vl), id, vl), vl);
        q               = __riscv_vmin_vx_i32m8(__riscv_vmax_vx_i32m8(q, -127, vl), 127, vl);
        __riscv_vse8_v_i8m2(line + c, __riscv_vncvt_x_x_w_i8m2(__riscv_vncvt_x_x_w_i16m4(q, vl), vl), vl);
        c += vl;
    }
    // One 16 B store per k-chunk: strided vlse64/vsse64 measured 2x slower here
    // (4.0 vs 1.9 us per 2048-wide row on one A100 core).
    for (int64_t j = 0; j < K / 16; ++j) {
        __riscv_vse8_v_i8m1(group + j * 256 + rr * 16, __riscv_vle8_v_i8m1(line + j * 16, 16), 16);
    }
}

static int64_t pad16(int64_t m) {
    return (m + 15) / 16 * 16;
}

// GGML_SPACEMIT_PERROW_PROF=1: per-worker phase times, printed at exit.
struct prof_slot {
    uint64_t ns[6];
    uint64_t calls;
    char     pad[8];
};
static prof_slot g_prof[64];
static const char * const g_prof_names[6] = { "quant", "sync", "a_copy", "b_copy", "tile", "epilogue" };

static bool prof_on() {
    static const bool on = [] {
        const char * env = std::getenv("GGML_SPACEMIT_PERROW_PROF");
        const bool   v   = env != nullptr && std::atoi(env) != 0;
        if (v) {
            std::atexit([] {
                prof_slot total = {};
                int       nw    = 0;
                for (const auto & p : g_prof) {
                    if (p.calls == 0) {
                        continue;
                    }
                    ++nw;
                    total.calls += p.calls;
                    for (int i = 0; i < 6; ++i) total.ns[i] += p.ns[i];
                }
                if (nw == 0) return;
                fprintf(stderr, "perrow prof: %d workers, %llu calls/worker, mean ms per worker:", nw,
                        (unsigned long long) (total.calls / nw));
                for (int i = 0; i < 6; ++i) fprintf(stderr, " %s=%.2f", g_prof_names[i], total.ns[i] / 1e6 / nw);
                fprintf(stderr, "\n");
            });
        }
        return v;
    }();
    return on;
}

static inline uint64_t prof_now() {
    timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t) t.tv_sec * 1000000000ull + (uint64_t) t.tv_nsec;
}

static size_t a_size(int64_t M, int64_t K) {
    return GGML_PAD((size_t) (pad16(M) * K + pad16(M) * (int64_t) sizeof(float)), 64);
}

// Quantized A lives in its own buffer rather than the shared workspace, so it
// survives the ops between two matmuls that read the same activation (Q/K/V
// after attn_norm, gate/up after ffn_norm; ROPE and others reuse the
// workspace). The second and later matmuls skip quantization and its barrier.
// The buffer is sized and the cache invalidated from work_size(), which the
// graph compute calls on the host thread before it launches the workers.
struct a_cache {
    int8_t *            buf  = nullptr;
    size_t              size = 0;
    const ggml_tensor * src  = nullptr;  // tensor whose quantized rows are in buf
    const void *        data = nullptr;
};
static a_cache g_a;

static void a_reserve(int64_t M, int64_t K) {
    const size_t need = a_size(M, K);
    if (need > g_a.size) {
        free(g_a.buf);
        g_a.buf  = static_cast<int8_t *>(aligned_alloc(64, need));
        g_a.size = g_a.buf ? need : 0;
        GGML_ASSERT(g_a.buf != nullptr);
    }
    g_a.src  = nullptr;
    g_a.data = nullptr;
}

// GGML_SPACEMIT_PERROW_CHECK=<calls>: for the first <calls> matmuls, worker 0
// recomputes three rows in scalar code and prints the kernel's error against the
// exact integer result of its own quantized inputs (kernel bugs, including the
// vector rotation, which the reference redoes with rotate_ref), and the error of
// the activation quantization against float activations with the stored weights.
template <typename XRow>
static void check(ggml::spacemit::context & ctx, const ggml_tensor * op, const XRow & x_row, const int8_t * w,
                  const float * ws, const int8_t * ap, const float * as) {
    static const int limit = [] {
        const char * env = std::getenv("GGML_SPACEMIT_PERROW_CHECK");
        return env ? std::atoi(env) : 0;
    }();
    static int calls = 0;
    if (limit <= 0) {
        return;
    }
    ctx.sync();
    if (ctx.ith == 0 && calls < limit) {
        ++calls;
        const ggml_tensor * src0 = op->src[0];
        const int64_t       K = src0->ne[0], N = src0->ne[1];
        const int64_t       M = op->src[1]->ne[1] * op->src[1]->ne[2] * op->src[1]->ne[3];
        const auto *        y = static_cast<const float *>(op->data);
        double              e_kern = 0, e_act = 0, ref2 = 0, outlier = 0;
        std::vector<float>  xr(K);
        for (int64_t r : { (int64_t) 0, M / 2, M - 1 }) {
            memcpy(xr.data(), x_row(r), K * sizeof(float));
            rotate_ref(xr.data(), K);
            double xmax = 0, x2 = 0;
            for (int64_t k = 0; k < K; ++k) {
                xmax = std::max(xmax, (double) std::fabs(xr[k]));
                x2 += (double) xr[k] * xr[k];
            }
            outlier = std::max(outlier, xmax / std::sqrt(x2 / K));
            for (int64_t n = 0; n < N; ++n) {
                const int8_t * col = w + (n / 32) * 32 * K + (n % 32) * 16;
                double         ref = 0;
                int64_t        acc = 0;
                for (int64_t k = 0; k < K; ++k) {
                    const int wq = col[(k / 16) * 512 + k % 16];
                    acc += (int64_t) wq * ap[(r / 16) * 16 * K + (k / 16) * 256 + (r % 16) * 16 + k % 16];
                    ref += (double) wq * xr[k];
                }
                const double exact = (double) acc * ws[n] * as[r];
                ref *= ws[n];
                const double out = y[r * N + n];
                e_kern += (out - exact) * (out - exact);
                e_act += (exact - ref) * (exact - ref);
                ref2 += ref * ref;
            }
        }
        fprintf(stderr, "perrow check %-24s M=%lld N=%lld K=%lld rot=%d kernel=%.2e act=%.3e outlier=%.1f\n",
                src0->name, (long long) M, (long long) N, (long long) K, rot_mode(), std::sqrt(e_kern / ref2),
                std::sqrt(e_act / ref2), outlier);
    }
    ctx.sync();
}

static bool forward_mul_mat(ggml::spacemit::context & ctx, ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];

    const int64_t K      = src0->ne[0];
    const int64_t N      = src0->ne[1];
    const int64_t M      = src1->ne[1] * src1->ne[2] * src1->ne[3];
    const int64_t M16    = pad16(M);
    const int64_t nstrip = (N + 31) / 32;
    const int     ith    = ctx.ith;
    const int     nth    = ctx.nth;

    GGML_ASSERT(src1->type == GGML_TYPE_F32 && src1->nb[0] == sizeof(float) && ggml_is_contiguous(op));
    GGML_ASSERT(g_a.size >= a_size(M, K));

    auto * tcm      = static_cast<uint8_t *>(ctx.shared.data);
    const size_t ts = ctx.shared.size;
    GGML_ASSERT(tcm != nullptr && ts >= (size_t) K * 5 && ts >= 48 * 64);

    const auto * w  = static_cast<const int8_t *>(src0->data);
    const auto * ws = reinterpret_cast<const float *>(w + nstrip * 32 * K);
    auto *       ap = g_a.buf;
    auto *       as = reinterpret_cast<float *>(ap + M16 * K);
    auto *       y  = static_cast<float *>(op->data);
    const auto   x_row = [src1](int64_t r) {
        const int64_t i11 = r % src1->ne[1];
        const int64_t i12 = (r / src1->ne[1]) % src1->ne[2];
        const int64_t i13 = r / (src1->ne[1] * src1->ne[2]);
        return reinterpret_cast<const float *>(static_cast<const char *>(src1->data) + i11 * src1->nb[1] +
                                               i12 * src1->nb[2] + i13 * src1->nb[3]);
    };

    const bool prof = prof_on();
    uint64_t   tp   = prof ? prof_now() : 0;
    auto       mark = [&](int i) {
        if (prof) {
            const uint64_t t = prof_now();
            g_prof[ith].ns[i] += t - tp;
            tp = t;
        }
    };
    if (prof) {
        g_prof[ith].calls++;
    }

    if (!(g_a.src == src1 && g_a.data == src1->data)) {
        // Phase 1: quantize A, four rows (one 64 B line per k-chunk) per unit,
        // unless the previous matmul already did it for this same activation.
        // All workers read the cache key here; it is only written below, after
        // every worker has passed this point (the barrier), and becomes visible
        // to the next op through the barrier that ends this one.
        const int64_t nunit = (M + 3) / 4;
        const int64_t u0    = nunit * ith / nth;
        const int64_t u1    = nunit * (ith + 1) / nth;
        for (int64_t r = u0 * 4; r < std::min(M, u1 * 4); ++r) {
            quantize_row(x_row(r), K, ap + (r / 16) * 16 * K, r % 16, as + r, tcm);
        }
        mark(0);
        ctx.sync();
        mark(1);
        if (ith == 0) {
            g_a.src  = src1;
            g_a.data = src1->data;
        }
    }

    // Phase 2: units are (column strip, 16-row tile) in strip-major order, split
    // evenly; a worker copies each of its weight strips into TCM once per k chunk
    // and runs every M tile it owns against it.
    alignas(128) int32_t tmp[16 * 32];
    int8_t *     tcmd  = reinterpret_cast<int8_t *>(tcm);
    const size_t avail = ts;

    const auto kc_for = [&](int64_t rows) { return std::min<int64_t>(K, (int64_t) (avail / (rows + 32)) / 64 * 64); };
    int64_t mpan = M16;
    while (mpan > 16 && kc_for(mpan) < std::min<int64_t>(K, 256)) {
        mpan = std::max<int64_t>(16, pad16(mpan / 2));
    }
    const int64_t kc = kc_for(mpan);
    GGML_ASSERT(kc >= 64);

    for (int64_t m0 = 0; m0 < M16; m0 += mpan) {
        const int64_t nmt = std::min(mpan, M16 - m0) / 16;
        const int64_t U   = nstrip * nmt;
        const int64_t u0  = U * ith / nth;
        const int64_t u1  = U * (ith + 1) / nth;
        if (u0 >= u1) {
            continue;
        }
        const int64_t s_first = u0 / nmt;
        const int64_t s_last  = (u1 - 1) / nmt;
        const int64_t t_lo    = s_first == s_last ? u0 % nmt : 0;
        const int64_t t_hi    = s_first == s_last ? (u1 - 1) % nmt + 1 : nmt;

        for (int64_t k0 = 0; k0 < K; k0 += kc) {
            const int64_t count = std::min(kc, K - k0);
            const bool    first = k0 == 0;
            const bool    last  = k0 + count == K;
            int8_t *      a     = tcmd;
            int8_t *      b     = tcmd + mpan * count;
            for (int64_t t = t_lo; t < t_hi; ++t) {
                spacemit_kernels::rvv::memcpy1d(a + t * 16 * count, ap + ((m0 / 16 + t) * 16 * K) + k0 * 16,
                                                count * 16);
            }
            mark(2);
            for (int64_t s = s_first; s <= s_last; ++s) {
                const int64_t ts0 = s == s_first ? u0 % nmt : 0;
                const int64_t ts1 = s == s_last ? (u1 - 1) % nmt + 1 : nmt;
                // Running the tile on the strip in DRAM instead was slower both at
                // M = 156 (-20%) and for decode (-7%).
                spacemit_kernels::rvv::memcpy1d(b, w + (s * K + k0) * 32, count * 32);
                mark(3);
                const int64_t n  = s * 32;
                const size_t  vl = (size_t) std::min<int64_t>(32, N - n);
                const auto    wsv = __riscv_vle32_v_f32m1(ws + n, vl);
                for (int64_t t = ts0; t < ts1; ++t) {
                    tile_m16n32(a + t * 16 * count, b, count / 16, tmp);
                    mark(4);
                    const int64_t r0 = m0 + t * 16;
                    for (int64_t rr = 0; rr < 16 && r0 + rr < M; ++rr) {
                        float * dst = y + (r0 + rr) * N + n;
                        auto    q   = __riscv_vle32_v_i32m1(tmp + rr * 32, vl);
                        if (!first) {
                            q = __riscv_vadd_vv_i32m1(
                                q, __riscv_vreinterpret_v_f32m1_i32m1(__riscv_vle32_v_f32m1(dst, vl)), vl);
                        }
                        if (last) {
                            auto sc = __riscv_vfmul_vf_f32m1(wsv, as[r0 + rr], vl);
                            __riscv_vse32_v_f32m1(dst, __riscv_vfmul_vv_f32m1(__riscv_vfcvt_f_x_v_f32m1(q, vl), sc, vl),
                                                  vl);
                        } else {
                            __riscv_vse32_v_f32m1(dst, __riscv_vreinterpret_v_i32m1_f32m1(q), vl);
                        }
                    }
                    mark(5);
                }
            }
        }
    }
    check(ctx, op, x_row, w, ws, ap, as);
    return true;
}

}  // namespace perrow_q8

class tensor_traits_q8_0_perrow : public ggml::spacemit::tensor_traits_base {
    bool work_size(int /* n_threads */, const ggml_tensor * op, size_t & size) const override {
        if (op->op != GGML_OP_MUL_MAT) {
            return false;
        }
        perrow_q8::a_reserve(op->src[1]->ne[1] * op->src[1]->ne[2] * op->src[1]->ne[3], op->src[0]->ne[0]);
        size = 0;
        return true;
    }

    bool compute_forward(ggml::spacemit::context & ctx, ggml_tensor * op) const override {
        if (op->op != GGML_OP_MUL_MAT) {
            return false;
        }
        return perrow_q8::forward_mul_mat(ctx, op);
    }

    int repack(ggml_tensor * t, const void * data, size_t data_size) const override {
        GGML_UNUSED(t);
        GGML_UNUSED(data);
        GGML_UNUSED(data_size);
        return -1;
    }
};

static const tensor_traits_q8_0_perrow q8_0_perrow;

// Q8_0 weights: per-row tensors are detected from their own scales at load time
// and moved to the per-row traits above; all others keep the 32x32 layout.
class tensor_traits_q8_0 : public tensor_traits<block_q8_0, 32, 32> {
    int repack(ggml_tensor * t, const void * data, size_t data_size) const override {
        if (perrow_q8::eligible(t, data, data_size)) {
            perrow_q8::repack(t, static_cast<const block_q8_0 *>(data));
            t->extra = (void *) static_cast<const ggml::spacemit::tensor_traits_base *>(&q8_0_perrow);
            GGML_LOG_DEBUG("%s: repack tensor %s as per-row INT8\n", __func__, t->name);
            return 0;
        }
        return ggml::cpu::riscv64_spacemit::repack<block_q8_0, 32, 32>(t, data, data_size);
    }
};

static const tensor_traits_q8_0 q8_0_32x32_q8_0_or_perrow;

// F16 weights under GGML_SPACEMIT_PERROW=2: quantized per row at load time.
class tensor_traits_f16_perrow : public tensor_traits_q8_0_perrow {
    int repack(ggml_tensor * t, const void * data, size_t data_size) const override {
        GGML_ASSERT(data_size == (size_t) (t->ne[0] * t->ne[1]) * sizeof(ggml_half));
        perrow_q8::repack_f16(t, static_cast<const ggml_half *>(data));
        t->extra = (void *) static_cast<const ggml::spacemit::tensor_traits_base *>(&q8_0_perrow);
        GGML_LOG_DEBUG("%s: repack tensor %s from F16 as per-row INT8\n", __func__, t->name);
        return 0;
    }
};

static const tensor_traits_f16_perrow f16_perrow;
#endif

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

// Q5_0 weights converted exactly to Q8_0 at load time and run on the Q8_0 IME path
class tensor_traits_q5_0_as_q8_0 : public tensor_traits<block_q8_0, 32, 32> {
    int repack(ggml_tensor * t, const void * data, size_t data_size) const override {
        return repack_q5_0_to_q8_0_32(t, data, data_size);
    }
};

static const tensor_traits_q5_0_as_q8_0 q5_0_as_q8_0_32x32;
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
                    return &ggml::cpu::riscv64_spacemit::q8_0_32x32_q8_0_or_perrow;
                }
#endif
            }
            break;
        case GGML_TYPE_F16:
            {
#if defined(RISCV64_SPACEMIT_IME2)
                if (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2 &&
                    ggml::cpu::riscv64_spacemit::perrow_q8::eligible_f16(cur)) {
                    return &ggml::cpu::riscv64_spacemit::f16_perrow;
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
                // exact conversion to Q8_0 at load time, run on the Q8_0 IME path
                if (cur->ne[1] % 32 == 0 && cur->ne[0] % QK8_0 == 0 &&
                    (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    return &ggml::cpu::riscv64_spacemit::q5_0_as_q8_0_32x32;
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
            nbytes = remap_block_nbytes(sizeof(block_q6_K), sizeof(block_q8_0) * 8, 32);
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

bool ggml_spacemit_compute_opnode(ggml::spacemit::context & ctx, const spacemit_opnode & node) {
    // Single-op opnodes with a special opcode: the two halves of the concat -> ssm_conv fusion.
    // They run at their original graph positions (see try_fuse_node); the allocator plans buffer
    // reuse on that order, so they must not be merged into one opnode at the concat position.
    switch (node.opcode) {
        case SPACEMIT_OP_CONCAT_STATE:
            // only the trailing columns read back through views (the conv state update)
            if (!spacemit_kernels::rvv::forward_concat_dim0_transposed_f32(ctx, node.node,
                                                                           (int64_t) node.kernel_params[0])) {
                GGML_ABORT("fused concat state update: unsupported shape");
            }
            return true;
        case SPACEMIT_OP_SSM_CONV_CONCAT:
            // the concat only materialized its trailing columns; read its inputs directly
            if (!spacemit_kernels::rvv::forward_ssm_conv_concat_f32(ctx, node.node)) {
                GGML_ABORT("fused ssm_conv: unsupported shape");
            }
            return true;
        default:
            break;
    }
    // Fused opnodes dispatch through the fused-op table; single-op opnodes keep
    // the existing traits path.
    if (!node.fused.empty()) {
        return ggml_spacemit_compute_fused(ctx, node);
    }
    return ggml_spacemit_compute_forward(ctx, node.node);
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
            // row-contiguous F32 copy with arbitrary outer strides (GDN rollback-snapshot copies)
            if (op->src[0] && op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32 &&
                op->nb[0] == sizeof(float) && op->src[0]->nb[0] == sizeof(float) &&
                op->ne[0] == op->src[0]->ne[0] && op->ne[1] == op->src[0]->ne[1] &&
                op->ne[2] == op->src[0]->ne[2] && op->ne[3] == op->src[0]->ne[3]) {
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
