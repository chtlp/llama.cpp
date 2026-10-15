#pragma once

#include "ggml-cpu-impl.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>

namespace spacemit_kernels {

template <typename T, typename U> constexpr auto div_round_up(T up, U down) {
    return (up + down - 1) / down;
}

// Q8 Blk [f32] [s16] [int8 * blk_len]
// Q8 Blk N [f32 * N] [s16 * N] [int8 * blk_len * N]
constexpr size_t q8_blk_size(size_t blk_len, bool with_blk_sum = false) {
    const size_t blk_size = sizeof(float) + blk_len * sizeof(int8_t) + (with_blk_sum ? sizeof(int16_t) : 0);
    return blk_size;
}

// Q8 HP row block: K is split into K32 subblocks.
// Each subblock stores [f32 scale] [int8 * 32], with an optional fp16 sum trailer per subblock.
constexpr size_t q8_hp_blk_size(size_t blk_len, bool with_blk_sum = false, bool with_blk_scale = false) {
    const size_t subblk_count = div_round_up(blk_len, size_t(32));
    const size_t blk_size     = blk_len * sizeof(int8_t) + subblk_count * sizeof(_Float16) +
                            (with_blk_sum ? subblk_count * sizeof(_Float16) : 0) +
                            (with_blk_scale ? sizeof(_Float16) : 0);
    return blk_size;
}

// Aligned 4-row Q8 HP layout used with signed-int4 Q4_0 weights (GGML_SPACEMIT_HP_ALIGNED, see
// hp_aligned_enabled()). Per K256 block of 4 rows:
//   [8 x 128B int8 4x32 subblock data][8 x 8B subblock scale headers (fp16 scale + pad)]
//   [72B trailer: fp16 a_sum[4][8] + fp16 scale_avg at +64][56B pad]                    = 1216B
// The int8 data must be 64B aligned: misaligned whole-register vector loads are 2.7x-7.7x slower on A100.
// The trailer sits at +1088 like the original layout; the original 1160B layout puts data at k*136+8.
constexpr size_t q8_hp_m4al_blk_size = 1216;
// per-row stride per K256 block when 4-row aligned groups and 1-row (q8_hp_blk_size) rows share a buffer
constexpr size_t q8_hp_al_row_blk_size = q8_hp_m4al_blk_size / 4;
constexpr size_t q8_hp_al_align        = 128;

// GGML_SPACEMIT_HP_ALIGNED=0 disables the aligned-A / signed-B Q4_0 hp path (default on). Read once, so the
// weight repack and the kernels always agree.
bool hp_aligned_enabled();

// Q8K Blk [f32] [s16 * (blk_len / 16)] [int8 * blk_len]
// Q8K Blk N [f32 * N] [s16 * (blk_len / 16) * N] [int8 * blk_len * N]
constexpr size_t q8k_blk_size(size_t blk_len) {
    const size_t blk_size = sizeof(float) + blk_len * sizeof(int8_t) + sizeof(int16_t) * blk_len / 16;
    return blk_size;
}

using quantize_a_row_def = std::function<void(size_t, const float *, size_t, uint8_t *)>;

namespace rvv {
void memcpy1d(void * dst, const void * src, int64_t size);

void memcpy2d(void * dst, int64_t dst_stride, const void * src, int64_t src_stride, int64_t tile_rows, int64_t size);

void forward_flash_attn_ext_f16_one_chunk_vlen1024_vf16(const ggml_compute_params * params,
                                                        ggml_tensor *               dst,
                                                        int                         ir0,
                                                        int                         ir1,
                                                        void *                      tcm_buffer,
                                                        size_t                      tcm_buffer_size);

// split-KV decode: returns KV chunks per KV head when applicable, else 0
int  flash_attn_ext_f16_split_kv_chunks_vlen1024_vf16(const ggml_tensor * dst, int nth);
void forward_flash_attn_ext_f16_split_kv_partial_vlen1024_vf16(const ggml_compute_params * params,
                                                               ggml_tensor *               dst,
                                                               int                         n_chunks,
                                                               float *                     partials,
                                                               void *                      tcm_buffer,
                                                               size_t                      tcm_buffer_size);
void forward_flash_attn_ext_f16_split_kv_reduce_vlen1024_vf16(const ggml_compute_params * params,
                                                              ggml_tensor *               dst,
                                                              int                         n_chunks,
                                                              const float *               partials);

void forward_flash_attn_ext_f16_tiled_vlen1024_vf16(const ggml_compute_params * params,
                                                    ggml_tensor *               dst,
                                                    int                         ir0,
                                                    int                         ir1,
                                                    void *                      tcm_buffer,
                                                    size_t                      tcm_buffer_size);

void forward_rms_norm_f32(ggml_compute_params * params, ggml_tensor * op);

void forward_norm_f32(ggml_compute_params * params, ggml_tensor * op);

void forward_cont_with_permute(ggml_compute_params * params, ggml_tensor * op);

void forward_cpy_with_permute(ggml_compute_params * params, ggml_tensor * op);

template <typename T> void forward_get_rows(ggml_compute_params * params, ggml_tensor * op);

template <typename T> void forward_concat(ggml_compute_params * params, ggml_tensor * op);

template <ggml_op op_type, typename T> void forward_binary(ggml_compute_params * params, ggml_tensor * op);

void forward_unary_tanh_f32(ggml_compute_params * params, ggml_tensor * op);
void forward_unary_gelu_f32(ggml_compute_params * params, ggml_tensor * op);

void forward_glu_geglu_f32(ggml_compute_params * params, ggml_tensor * op);

template <typename T> void forward_sum_rows(const ggml_compute_params * params, ggml_tensor * op);

bool forward_concat_dim0_transposed_f32(const ggml_compute_params * params, ggml_tensor * op);

bool forward_ssm_conv_f32(const ggml_compute_params * params, ggml_tensor * dst);

bool forward_gated_delta_net_f32(const ggml_compute_params * params, ggml_tensor * dst);

template <typename T> void forward_repeat_nrows(ggml_compute_params * params, ggml_tensor * op);

template <typename T> void forward_repeat_dim1(ggml_compute_params * params, ggml_tensor * op);

void quantize_a_row_i8(size_t blk_len, const float * a_ptr, size_t count_k, uint8_t * quant_a_ptr);

void quantize_a_4row_i8(size_t blk_len, const float * a_ptr, size_t count_k, uint8_t * quant_a_ptr);

void quantize_a_row_i8_hp(size_t blk_len, const float * a_ptr, size_t count_k, uint8_t * quant_a_ptr);

void quantize_a_4row_i8_hp(size_t blk_len, const float * a_ptr, size_t count_k, uint8_t * quant_a_ptr);

// aligned 4-row layout (q8_hp_m4al_blk_size per K256 block)
void quantize_a_4row_i8_hp_al(size_t blk_len, const float * a_ptr, size_t count_k, uint8_t * quant_a_ptr);

void quantize_a_row_i8k(size_t blk_len, const float * a_ptr, size_t count_k, uint8_t * quant_a_ptr);

void quantize_a_4row_i8k(size_t blk_len, const float * a_ptr, size_t count_k, uint8_t * quant_a_ptr);

}  // namespace rvv

}  // namespace spacemit_kernels
