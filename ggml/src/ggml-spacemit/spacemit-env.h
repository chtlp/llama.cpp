#pragma once

#include "spine_barrier.h"
#include "spine_mem_pool.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <vector>

namespace ggml::cpu::riscv64_spacemit {

inline constexpr size_t spine_init_barrier_count = 16;

enum class spine_core_arch_id : uint16_t {
    core_arch_none = 0,
    core_arch_x60  = 0x503C,
    core_arch_x100 = 0x5064,
    core_arch_x200 = 0x50C8,
    core_arch_a60  = 0xA03C,
    core_arch_a100 = 0xA064,
    core_arch_a200 = 0xA0C8,
};

struct spine_env_info {
    int                    num_cores{ 0 };
    size_t                 vlen{ 0 };
    spine_core_arch_id     perfer_core_arch_id{ spine_core_arch_id::core_arch_none };
    bool                   use_ime2{ false };
    bool                   use_ime1{ false };
    spine_mem_pool_backend mem_backend{ spine_mem_pool_backend::transparent_hugepage };
    spine_barrier_t *      init_barrier{ nullptr };
    std::vector<int>       perfer_core_ids;

    spine_env_info();
    ~spine_env_info();
};

extern spine_env_info global_spine_env_info;

struct spacemit_opt_flags {
    bool    q41_m4          = true;
    int64_t q41_m4_max      = 8;
    bool    q8_small_m      = true;
    bool    moe_small_m     = true;
    int64_t moe_pad8_min_m  = 32;
    bool    gdn_mtp         = true;
    bool    fa_splitkv      = true;
    int64_t fa_ime2_min_q   = 16;
    bool    fa_m1_ime2      = true;
    bool    fa_ime2_stage   = true;
    bool    fa_prefill_ime2 = true;
    bool    fa_tiled_nozero = true;
    bool    mtp_head_q4     = true;
    int64_t mtp_head_vocab  = 0;
    bool    arange_cast     = true;

    static bool read_env_bool(const char * name, bool def_val) {
        const char * v = std::getenv(name);
        return v == nullptr ? def_val : (std::atoi(v) != 0);
    }

    static int64_t read_env_i64(const char * name, int64_t def_val) {
        const char * v = std::getenv(name);
        return v == nullptr ? def_val : std::atoll(v);
    }

    spacemit_opt_flags() {
        q41_m4          = read_env_bool("GGML_SPACEMIT_Q41_M4", true);
        q41_m4_max      = read_env_i64("GGML_SPACEMIT_Q41_M4_MAX", 8);
        if (q41_m4_max < 0) q41_m4_max = 0;
        if (q41_m4_max > 16) q41_m4_max = 16;
        q8_small_m      = read_env_bool("GGML_SPACEMIT_Q8_SMALLM", true);
        moe_small_m     = read_env_bool("GGML_SPACEMIT_MOE_SMALLM", true);
        moe_pad8_min_m  = read_env_i64("GGML_SPACEMIT_MOE_PAD8_MIN_M", 32);
        gdn_mtp         = read_env_bool("GGML_SPACEMIT_GDN_MTP", true);
        fa_splitkv      = read_env_bool("GGML_SPACEMIT_FA_SPLITKV_PAIR", true);
        fa_ime2_min_q   = read_env_i64("GGML_SPACEMIT_FA_IME2_MIN_Q", 16);
        fa_m1_ime2      = read_env_bool("GGML_SPACEMIT_FA_M1_IME2", true);
        fa_ime2_stage   = read_env_bool("GGML_SPACEMIT_FA_IME2_STAGE", true);
        fa_prefill_ime2 = read_env_bool("GGML_SPACEMIT_FA_PREFILL_IME2", true);
        fa_tiled_nozero = read_env_bool("GGML_SPACEMIT_FA_TILED_NOZERO", true);
        mtp_head_q4     = read_env_bool("GGML_SPACEMIT_MTP_HEAD_Q4", true);
        mtp_head_vocab  = read_env_i64("GGML_SPACEMIT_MTP_HEAD_VOCAB", 0);
        arange_cast     = read_env_bool("GGML_SPACEMIT_ARANGE_CAST", true);
    }

    static spacemit_opt_flags & instance() {
        static spacemit_opt_flags inst;
        return inst;
    }
};

}  // namespace ggml::cpu::riscv64_spacemit
