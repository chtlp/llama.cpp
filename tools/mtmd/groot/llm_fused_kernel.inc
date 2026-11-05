// K3 single-block prefill, feature-last FP16 input/output, no historical KV.
// Packed weights are borrowed. All input-dependent packing, temporary data,
// DDR/TCM copies and output writes occur on every run.
// Public interfaces: Matrix, BlockParameters, PrefillBlock; BF16 conversion,
// RMSNorm and GEMM building blocks. See README for numeric policies and limits.
#include <algorithm>
#include <atomic>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <exception>
#include <fstream>
#include <memory>
#include <pthread.h>
#include <riscv_vector.h>
#include <sched.h>
#include <stdexcept>
#include <string>
#include <thread>
#include <time.h>
#include <unistd.h>
#include <vector>
namespace nvfp4_block {
// Cache-line aligned DDR panels avoid split vector transactions at every copy.
template <class T> struct AlignedAllocator {
  using value_type = T;
  AlignedAllocator() = default;
  template <class U> AlignedAllocator(const AlignedAllocator<U> &) {}
  T *allocate(size_t n) {
    if (n > SIZE_MAX / sizeof(T))
      throw std::bad_alloc();
    void *p = nullptr;
    if (posix_memalign(&p, 128, std::max(size_t(128), n * sizeof(T))))
      throw std::bad_alloc();
    return (T *)p;
  }
  void deallocate(T *p, size_t) { free(p); }
  template <class U> bool operator==(const AlignedAllocator<U> &) const { return true; }
  template <class U> bool operator!=(const AlignedAllocator<U> &) const { return false; }
};
template <class T> using AlignedVector = std::vector<T, AlignedAllocator<T>>;
namespace fp4fp8 {
static uint64_t ns() {
  timespec t;
  clock_gettime(CLOCK_MONOTONIC_RAW, &t);
  return uint64_t(t.tv_sec) * 1000000000 + t.tv_nsec;
}
} // namespace fp4fp8
namespace fp4_residual {
inline void copy_gu_input(void *dst, const void *src, size_t bytes) {
#if defined(__riscv)
  auto *d = static_cast<uint8_t *>(dst);
  auto *s = static_cast<const uint8_t *>(src);
  size_t blocks = bytes / 512;
  if (blocks)
    asm volatile("1:\n\t"
                 "vl4r.v v0,(%[s])\n\t"
                 "vs4r.v v0,(%[d])\n\t"
                 "addi %[s],%[s],512\n\t"
                 "addi %[d],%[d],512\n\t"
                 "addi %[n],%[n],-1\n\t"
                 "bnez %[n],1b\n\t"
                 : [s] "+&r"(s), [d] "+&r"(d), [n] "+&r"(blocks)
                 :
                 : "memory", "v0", "v1", "v2", "v3");
  memcpy(d, s, bytes % 512);
#else
  memcpy(dst, src, bytes);
#endif
}
} // namespace fp4_residual
#ifndef SPINE_TCM_PUBLIC_H_
#define SPINE_TCM_PUBLIC_H_

/*
 * spine_tcm public API
 *
 * Usage:
 *   1. Direct link mode
 *      Define SPINE_TCM_DIRECT_LINK and link against libspine_tcm.so.
 *
 *      if (spine_tcm_is_available()) {
 *          void *buffer = spine_tcm_mem_get(0);
 *          spine_tcm_mem_free(0);
 *      }
 *
 *   2. Header-only loader mode
 *      Include this header without linking libspine_tcm.so. The loader first
 *      tries to reuse a process-global spine_tcm instance and falls back to
 *      dlopen("libspine_tcm.so") when needed.
 *
 *      spine_tcm_open_handle(NULL);  // optional pre-bind
 *      if (spine_tcm_is_available()) {
 *          void *buffer = spine_tcm_mem_get(0);
 *          spine_tcm_mem_free(0);
 *      }
 */

#if !defined(SPINE_TCM_BUILD_SHARED) && !defined(SPINE_TCM_DIRECT_LINK)
#include <dlfcn.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#if defined(SPINE_TCM_BUILD_SHARED)
#define SPINE_TCM_API __declspec(dllexport)
#else
#define SPINE_TCM_API __declspec(dllimport)
#endif
#else
#define SPINE_TCM_API __attribute__((visibility("default")))
#endif

typedef struct spine_tcm_mem_info {
  size_t blk_size;
  size_t blk_num;
  int is_fake_tcm;
} spine_tcm_mem_info_t;

typedef struct spine_tcm_block_info {
  int id;
  void *va;
  size_t size;
  uint64_t phys_addr;
  uint64_t cpu_affinity_mask;
  int owner_tid;
  int is_acquired;
} spine_tcm_block_info_t;

/* Shared-library runtime ABI exported by libspine_tcm.so. */
SPINE_TCM_API const char *spine_tcm_runtime_version(void);
SPINE_TCM_API int spine_tcm_runtime_is_available(void);
SPINE_TCM_API int spine_tcm_runtime_layout_info(spine_tcm_mem_info_t *info);
SPINE_TCM_API int spine_tcm_runtime_mem_info(int id, spine_tcm_block_info_t *info);
SPINE_TCM_API void *spine_tcm_runtime_mem_get(int id);
SPINE_TCM_API int spine_tcm_runtime_mem_free(int id);
SPINE_TCM_API void *spine_tcm_runtime_mem_try_wait(int id, size_t timeout_us);
SPINE_TCM_API int spine_tcm_runtime_mem_release(int id);
SPINE_TCM_API int spine_tcm_runtime_mem_force_release(int id);
SPINE_TCM_API int spine_tcm_runtime_mem_query(int id);

#if defined(SPINE_TCM_DIRECT_LINK)
/* Optional no-op in direct-link mode. */
static inline int spine_tcm_open_handle(const char *so_path) {
  (void)so_path;
  return 0;
}

static inline const char *spine_tcm_version(void) { return spine_tcm_runtime_version(); }

/* Returns 1 when the runtime driver is available, otherwise 0. */
static inline int spine_tcm_is_available(void) { return spine_tcm_runtime_is_available(); }

/* Returns runtime memory geometry and whether the current backend is fake TCM.
 */
static inline int spine_tcm_mem_info(spine_tcm_mem_info_t *info) {
  return spine_tcm_runtime_layout_info(info);
}

/* Returns per-block runtime metadata for the given TCM id. */
static inline int spine_tcm_block_info(int id, spine_tcm_block_info_t *info) {
  return spine_tcm_runtime_mem_info(id, info);
}

/* Returns a cached buffer for the given TCM id, or NULL on failure. */
static inline void *spine_tcm_mem_get(int id) { return spine_tcm_runtime_mem_get(id); }

/* Releases one reference acquired by spine_tcm_mem_get(id). */
static inline int spine_tcm_mem_free(int id) { return spine_tcm_runtime_mem_free(id); }

/* Waits for a TCM block handoff and returns the driver-owned buffer when
 * available. */
static inline void *spine_tcm_mem_try_wait(int id, size_t over_time) {
  return spine_tcm_runtime_mem_try_wait(id, over_time);
}

/* Releases a buffer acquired by spine_tcm_mem_try_wait(id, over_time). */
static inline int spine_tcm_mem_release(int id) { return spine_tcm_runtime_mem_release(id); }

/* Forces a release for the given TCM id when the backend supports it. */
static inline int spine_tcm_mem_force_release(int id) {
  return spine_tcm_runtime_mem_force_release(id);
}

/* Returns whether the given TCM id is currently acquired. */
static inline int spine_tcm_mem_query(int id) { return spine_tcm_runtime_mem_query(id); }
#elif !defined(SPINE_TCM_BUILD_SHARED)
typedef struct spine_tcm_handle {
  void *module_handle;
  int use_global_scope;
  int owns_module_handle;
  const char *(*runtime_version)(void);
  int (*runtime_is_available)(void);
  int (*runtime_layout_info)(spine_tcm_mem_info_t *info);
  int (*runtime_mem_info)(int id, spine_tcm_block_info_t *info);
  void *(*runtime_mem_get)(int id);
  int (*runtime_mem_free)(int id);
  void *(*runtime_mem_try_wait)(int id, size_t over_time);
  int (*runtime_mem_release)(int id);
  int (*runtime_mem_force_release)(int id);
  int (*runtime_mem_query)(int id);
} spine_tcm_handle_t;

static inline spine_tcm_handle_t *spine_tcm_default_handle(void) {
  static spine_tcm_handle_t handle = {0};
  return &handle;
}

static inline void spine_tcm_handle_reset(spine_tcm_handle_t *handle) {
  if (handle != NULL) {
    memset(handle, 0, sizeof(*handle));
  }
}

static inline int spine_tcm_handle_bind(spine_tcm_handle_t *handle) {
  void *symbol_scope = handle->use_global_scope ? RTLD_DEFAULT : handle->module_handle;

  handle->runtime_version = (const char *(*)(void))dlsym(symbol_scope, "spine_tcm_runtime_version");
  handle->runtime_is_available =
      (int (*)(void))dlsym(symbol_scope, "spine_tcm_runtime_is_available");
  handle->runtime_layout_info =
      (int (*)(spine_tcm_mem_info_t *))dlsym(symbol_scope, "spine_tcm_runtime_layout_info");
  handle->runtime_mem_info =
      (int (*)(int, spine_tcm_block_info_t *))dlsym(symbol_scope, "spine_tcm_runtime_mem_info");
  handle->runtime_mem_get = (void *(*)(int))dlsym(symbol_scope, "spine_tcm_runtime_mem_get");
  handle->runtime_mem_free = (int (*)(int))dlsym(symbol_scope, "spine_tcm_runtime_mem_free");
  handle->runtime_mem_try_wait =
      (void *(*)(int, size_t))dlsym(symbol_scope, "spine_tcm_runtime_mem_try_wait");
  handle->runtime_mem_release = (int (*)(int))dlsym(symbol_scope, "spine_tcm_runtime_mem_release");
  handle->runtime_mem_force_release =
      (int (*)(int))dlsym(symbol_scope, "spine_tcm_runtime_mem_force_release");
  handle->runtime_mem_query = (int (*)(int))dlsym(symbol_scope, "spine_tcm_runtime_mem_query");

  return handle->runtime_version != NULL && handle->runtime_is_available != NULL &&
                 handle->runtime_layout_info != NULL && handle->runtime_mem_info != NULL &&
                 handle->runtime_mem_get != NULL && handle->runtime_mem_free != NULL &&
                 handle->runtime_mem_try_wait != NULL && handle->runtime_mem_release != NULL &&
                 handle->runtime_mem_force_release != NULL && handle->runtime_mem_query != NULL
             ? 0
             : -1;
}

/*
 * Try to bind against an already-loaded process-global spine_tcm instance.
 * The shared library exports spine_tcm_runtime_marker only for this probe.
 */
static inline int spine_tcm_try_bind_global(spine_tcm_handle_t *handle) {
  if (dlsym(RTLD_DEFAULT, "spine_tcm_runtime_marker") == NULL) {
    return -1;
  }

  handle->use_global_scope = 1;
  return spine_tcm_handle_bind(handle);
}

/*
 * Optional pre-bind entry point.
 *
 * Behavior:
 *   - Reuses an already-loaded global spine_tcm instance when available.
 *   - Otherwise loads the shared library from so_path or the default soname.
 *   - Repeated calls are safe and return 0 after the first successful bind.
 */
static inline int spine_tcm_open_handle(const char *so_path) {
  spine_tcm_handle_t *resolved = spine_tcm_default_handle();
  const char *library = (so_path != NULL && so_path[0] != '\0') ? so_path : "libspine_tcm.so";

  if (resolved->module_handle != NULL || resolved->use_global_scope) {
    return 0;
  }

  if (spine_tcm_try_bind_global(resolved) == 0) {
    return 0;
  }

  spine_tcm_handle_reset(resolved);

  resolved->module_handle = dlopen(library, RTLD_LAZY | RTLD_GLOBAL);
  resolved->owns_module_handle = resolved->module_handle != NULL ? 1 : 0;

  if (resolved->module_handle == NULL) {
    spine_tcm_handle_reset(resolved);
    return -1;
  }

  if (spine_tcm_handle_bind(resolved) != 0) {
    if (resolved->owns_module_handle) {
      dlclose(resolved->module_handle);
    }
    spine_tcm_handle_reset(resolved);
    return -1;
  }

  return 0;
}

/* Returns 1 when the runtime driver is available, otherwise 0. */
static inline int spine_tcm_is_available(void) {
  spine_tcm_handle_t *resolved = spine_tcm_default_handle();

  if (resolved->module_handle == NULL && !resolved->use_global_scope) {
    (void)spine_tcm_open_handle(NULL);
  }

  if ((resolved->module_handle == NULL && !resolved->use_global_scope) ||
      resolved->runtime_is_available == NULL) {
    return 0;
  }

  return resolved->runtime_is_available();
}

/* Returns runtime memory geometry and whether the current backend is fake TCM.
 */
static inline int spine_tcm_mem_info(spine_tcm_mem_info_t *info) {
  spine_tcm_handle_t *resolved = spine_tcm_default_handle();

  if (resolved->module_handle == NULL && !resolved->use_global_scope) {
    (void)spine_tcm_open_handle(NULL);
  }

  if ((resolved->module_handle == NULL && !resolved->use_global_scope) ||
      resolved->runtime_layout_info == NULL) {
    return -1;
  }

  return resolved->runtime_layout_info(info);
}

static inline const char *spine_tcm_version(void) {
  spine_tcm_handle_t *resolved = spine_tcm_default_handle();

  if (resolved->module_handle == NULL && !resolved->use_global_scope) {
    (void)spine_tcm_open_handle(NULL);
  }

  if ((resolved->module_handle == NULL && !resolved->use_global_scope) ||
      resolved->runtime_version == NULL) {
    return "unknown";
  }

  return resolved->runtime_version();
}

/* Returns per-block runtime metadata for the given TCM id. */
static inline int spine_tcm_block_info(int id, spine_tcm_block_info_t *info) {
  spine_tcm_handle_t *resolved = spine_tcm_default_handle();

  if (resolved->module_handle == NULL && !resolved->use_global_scope) {
    (void)spine_tcm_open_handle(NULL);
  }

  if ((resolved->module_handle == NULL && !resolved->use_global_scope) ||
      resolved->runtime_mem_info == NULL) {
    return -1;
  }

  return resolved->runtime_mem_info(id, info);
}

/* Returns a cached buffer for the given TCM id, or NULL on failure. */
static inline void *spine_tcm_mem_get(int id) {
  spine_tcm_handle_t *resolved = spine_tcm_default_handle();

  if (resolved->module_handle == NULL && !resolved->use_global_scope) {
    (void)spine_tcm_open_handle(NULL);
  }

  if (resolved->module_handle == NULL && !resolved->use_global_scope) {
    return NULL;
  }

  if (resolved->runtime_mem_get == NULL) {
    return NULL;
  }

  return resolved->runtime_mem_get(id);
}

/* Releases one reference acquired by spine_tcm_mem_get(id). */
static inline int spine_tcm_mem_free(int id) {
  spine_tcm_handle_t *resolved = spine_tcm_default_handle();

  if (resolved->module_handle == NULL && !resolved->use_global_scope) {
    (void)spine_tcm_open_handle(NULL);
  }

  if ((resolved->module_handle == NULL && !resolved->use_global_scope) ||
      resolved->runtime_mem_free == NULL) {
    return -1;
  }

  return resolved->runtime_mem_free(id);
}

/* Waits for a TCM block handoff and returns the driver-owned buffer when
 * available. */
static inline void *spine_tcm_mem_try_wait(int id, size_t over_time) {
  spine_tcm_handle_t *resolved = spine_tcm_default_handle();

  if (resolved->module_handle == NULL && !resolved->use_global_scope) {
    (void)spine_tcm_open_handle(NULL);
  }

  if (resolved->module_handle == NULL && !resolved->use_global_scope) {
    return NULL;
  }

  if (resolved->runtime_mem_try_wait == NULL) {
    return NULL;
  }

  return resolved->runtime_mem_try_wait(id, over_time);
}

/* Releases a buffer acquired by spine_tcm_mem_try_wait(id, over_time). */
static inline int spine_tcm_mem_release(int id) {
  spine_tcm_handle_t *resolved = spine_tcm_default_handle();

  if (resolved->module_handle == NULL && !resolved->use_global_scope) {
    (void)spine_tcm_open_handle(NULL);
  }

  if ((resolved->module_handle == NULL && !resolved->use_global_scope) ||
      resolved->runtime_mem_release == NULL) {
    return -1;
  }

  return resolved->runtime_mem_release(id);
}

/* Forces a release for the given TCM id when the backend supports it. */
static inline int spine_tcm_mem_force_release(int id) {
  spine_tcm_handle_t *resolved = spine_tcm_default_handle();

  if (resolved->module_handle == NULL && !resolved->use_global_scope) {
    (void)spine_tcm_open_handle(NULL);
  }

  if ((resolved->module_handle == NULL && !resolved->use_global_scope) ||
      resolved->runtime_mem_force_release == NULL) {
    return -1;
  }

  return resolved->runtime_mem_force_release(id);
}

/* Returns whether the given TCM id is currently acquired. */
static inline int spine_tcm_mem_query(int id) {
  spine_tcm_handle_t *resolved = spine_tcm_default_handle();

  if (resolved->module_handle == NULL && !resolved->use_global_scope) {
    (void)spine_tcm_open_handle(NULL);
  }

  if ((resolved->module_handle == NULL && !resolved->use_global_scope) ||
      resolved->runtime_mem_query == NULL) {
    return -1;
  }

  return resolved->runtime_mem_query(id);
}
#else
static inline const char *spine_tcm_version(void) { return spine_tcm_runtime_version(); }
#endif

#define SPINE_TCM_VERSION (spine_tcm_version())

#ifdef __cplusplus
}
#endif

#endif

namespace ime_i4_microkernel {

static void pack_pair(vint32m1_t q, uint8_t *low, uint8_t *high) {
  auto q8 = __riscv_vncvt_x_x_w_i8mf4(__riscv_vncvt_x_x_w_i16mf2(q, 32), 32);
  auto pairs = __riscv_vreinterpret_v_u8mf4_u16mf4(__riscv_vreinterpret_v_i8mf4_u8mf4(q8));
  auto lo = __riscv_vor_vv_u16mf4(
      __riscv_vand_vx_u16mf4(pairs, 15, 16),
      __riscv_vand_vx_u16mf4(__riscv_vsrl_vx_u16mf4(pairs, 4, 16), 240, 16), 16);
  auto hi = __riscv_vor_vv_u16mf4(
      __riscv_vand_vx_u16mf4(__riscv_vsrl_vx_u16mf4(pairs, 4, 16), 15, 16),
      __riscv_vand_vx_u16mf4(__riscv_vsrl_vx_u16mf4(pairs, 8, 16), 240, 16), 16);
  __riscv_vse8_v_u8mf8(low, __riscv_vncvt_x_x_w_u8mf8(lo, 16), 16);
  __riscv_vse8_v_u8mf8(high, __riscv_vncvt_x_x_w_u8mf8(hi, 16), 16);
}

static void pack_nibbles(vint32m1_t q, uint8_t *dst) {
  auto q8 = __riscv_vncvt_x_x_w_i8mf4(__riscv_vncvt_x_x_w_i16mf2(q, 32), 32);
  auto pairs = __riscv_vreinterpret_v_u8mf4_u16mf4(__riscv_vreinterpret_v_i8mf4_u8mf4(q8));
  auto packed = __riscv_vor_vv_u16mf4(
      __riscv_vand_vx_u16mf4(pairs, 15, 16),
      __riscv_vand_vx_u16mf4(__riscv_vsrl_vx_u16mf4(pairs, 4, 16), 240, 16), 16);
  __riscv_vse8_v_u8mf8(dst, __riscv_vncvt_x_x_w_u8mf8(packed, 16), 16);
}

static bool prepare(const float *input, int k, int group, uint8_t *dst, int components = 1,
                    bool residual_a4 = false, bool radix = false, bool check_range = false,
                    int scale_group = 0, int valid_rows = 8) {
  if (!scale_group)
    scale_group = group;
  const size_t stride = group * 8 + 256;
  const size_t pair_stride = radix         ? group * 12 + 256
                             : residual_a4 ? group * 12 + 512
                                           : components * stride;
  if (valid_rows < 8)
    std::fill_n(dst, size_t(k / group) * pair_stride, uint8_t(0));
  for (int row = 0; row < valid_rows; ++row)
    for (int start = 0; start < k; start += group) {
      const float *src = input + size_t(row) * k + start;
      auto maximum = __riscv_vfmv_v_f_f32m4(0, 128);
      for (int z = 0; z < group;) {
        size_t vl = __riscv_vsetvl_e32m4(group - z);
        const auto x = __riscv_vle32_v_f32m4(src + z, vl);
        if (check_range && __riscv_vcpop_m_b8(__riscv_vmfne_vv_f32m4_b8(x, x, vl), vl))
          return false;
        maximum = __riscv_vfmax_vv_f32m4_tu(maximum, maximum, __riscv_vfabs_v_f32m4(x, vl), vl);
        z += vl;
      }
      const float max_abs = __riscv_vfmv_f_s_f32m1_f32(
          __riscv_vfredmax_vs_f32m4_f32m1(maximum, __riscv_vfmv_v_f_f32m1(0, 1), 128));
      const float scale = max_abs ? max_abs / 127.0f : 1.0f;
      if (check_range &&
          (!std::isfinite(max_abs) || max_abs > 65000 || scale / float(group * 8) == 0))
        return false;
      const float inverse = group >= 64 ? 1.0f / scale : 0;
      const bool use_inverse = group >= 64 && std::isfinite(inverse);
      uint8_t *a = dst + size_t(start / group) * pair_stride;
      for (int z = 0; z < group; z += 32) {
        auto value = __riscv_vle32_v_f32m1(src + z, 32);
        auto x = use_inverse ? __riscv_vfmul_vf_f32m1(value, inverse, 32)
                             : __riscv_vfdiv_vf_f32m1(value, scale, 32);
        auto q = __riscv_vfcvt_x_f_v_i32m1(x, 32);
        q = __riscv_vmin_vx_i32m1(__riscv_vmax_vx_i32m1(q, -127, 32), 127, 32);
        uint8_t *block = a + (z / 32) * (radix ? 384 : 256);
        pack_pair(q, block + row * 16, block + 128 + row * 16);
        if (residual_a4) {
          auto value = __riscv_vfmul_vf_f32m1(__riscv_vfcvt_f_x_v_f32m1(q, 32), 0.0625f, 32);
          auto coarse = __riscv_vfcvt_x_f_v_i32m1(value, 32);
          coarse = __riscv_vmin_vx_i32m1(__riscv_vmax_vx_i32m1(coarse, -8, 32), 7, 32);
          pack_nibbles(coarse,
                       radix ? block + 256 + row * 16 : a + stride + (z / 32) * 128 + row * 16);
        }
      }
      std::fill_n(reinterpret_cast<float *>(a + group * (radix ? 12 : 8)) + row * 8, 8,
                  scale / float(scale_group * 8));
      if (residual_a4 && !radix)
        std::fill_n(reinterpret_cast<float *>(a + stride + group * 4) + row * 8, 8,
                    scale / float(group * 8));
    }
  for (int g = 0; !residual_a4 && g < k / group; ++g)
    for (int c = 1; c < components; ++c) {
      auto *a = dst + size_t(g) * components * stride;
      fp4_residual::copy_gu_input(a + c * stride, a, stride);
    }
  return true;
}

template <int Diagnostic = 0, bool ResidualA4 = false, bool RawAccumulator = false,
          bool Sparse = false, bool Radix = false, bool SplitActivation = false>
__attribute__((noinline)) static void
tile_impl(const uint8_t *a, const uint8_t *b, int groups, int subblocks, float *output,
          size_t stride, bool accumulate = false, const uint8_t *active = nullptr) {
  // High bits remove stages without changing the low-bit instruction schedule.
  static_assert(!SplitActivation || (Radix && (Diagnostic & 7) == 5 && !ResidualA4 && !Sparse));
  static_assert(!(Diagnostic & 192) || Radix);
  static_assert(!(Diagnostic & 256) || (Radix && (Diagnostic & 7) == 5));
  static_assert(!(Diagnostic & 512) || (Radix && (Diagnostic & 7) == 5));
  static_assert(!(Diagnostic & 1024) || (Radix && (Diagnostic & 7) == 5));
  static_assert(!(Diagnostic & 2048) || (Diagnostic & 1024));
  asm volatile(
      "vsetvli t0, zero, e32, m8, ta, ma\n\t"
      ".if %c[raw]\n\t"
      "beqz %[accumulate], 8f\n\t"
      "vl8r.v v24, (%[out])\n\t"
      "j 9f\n\t"
      "8:\n\t"
      ".endif\n\t"
      "vmv.v.i v24, 0\n\t"
      "9:\n\t"
      "li t3, 0\n\t"
      ".if %c[split_a]\n\t"
      "li t5, 0\n\t"
      ".endif\n\t"
      ".if %c[noload]\n\t"
      "vmv.v.i v0, 0\n\t"
      ".endif\n\t"
      "1:\n\t"
      ".if %c[sparse]\n\t"
      "lbu t4, 0(%[active])\n\t"
      "addi %[active], %[active], 1\n\t"
      "bnez t4, 6f\n\t"
      "slli t4, %[sub], 8\n\t"
      "beqz t3, 11f\n\t"
      "srli t4, t4, 1\n\t"
      "11:\n\t"
      "add %[a], %[a], t4\n\t"
      "addi %[a], %[a], 256\n\t"
      "j 10f\n\t"
      "6:\n\t"
      ".endif\n\t"
      "vmv.v.i v8, 0\n\t"
      "vmv.v.i v16, 0\n\t"
      "mv t2, %[sub]\n\t"
      ".if %c[split_a]\n\t"
      "srli t2, t2, 1\n\t"
      ".endif\n\t"
      "vsetvli t0, zero, e8, m1, ta, ma\n\t"
      ".if %c[mixed]\n\t"
      "bnez t3, 12f\n\t"
      ".endif\n\t"
      ".if %c[preload_a] && %c[load]\n\t"
      "vl2r.v v0, (%[a])\n\t"
      ".endif\n\t"
      "2:\n\t"
      ".if %c[load]\n\t"
      ".if !%c[preload_a]\n\t"
      "vl2r.v v0, (%[a])\n\t"
      ".endif\n\t"
      "vl4r.v v4, (%[b])\n\t"
      ".if %c[radix] && !%c[late_coarse]\n\t"
      "addi t4, %[a], 256\n\t"
      "vl1r.v v2, (t4)\n\t"
      ".endif\n\t"
      ".endif\n\t"
      ".if %c[dot]\n\t"
      "vmadotus v8, v0, v4, i4\n\t"
      "vmadotus v10, v0, v5, i4\n\t"
      "vmadotus v12, v0, v6, i4\n\t"
      "vmadotus v14, v0, v7, i4\n\t"
      ".endif\n\t"
      ".if %c[late_coarse] && %c[load]\n\t"
      "addi t4, %[a], 256\n\t"
      "vl1r.v v2, (t4)\n\t"
      ".endif\n\t"
      ".if %c[dot]\n\t"
      "3:\n\t"
      "vmadot v16, v1, v4, i4\n\t"
      "vmadot v18, v1, v5, i4\n\t"
      "vmadot v20, v1, v6, i4\n\t"
      "vmadot v22, v1, v7, i4\n\t"
      ".endif\n\t"
      ".if %c[radix]\n\t"
      ".if %c[load]\n\t"
      "addi t4, %[b], 512\n\t"
      "vl4r.v v4, (t4)\n\t"
      ".endif\n\t"
      ".if %c[preload_a]\n\t"
      "addi %[a], %[a], 128\n\t"
      "addi %[a], %[a], 256\n\t"
      ".if %c[load]\n\t"
      "vl2r.v v0, (%[a])\n\t"
      ".endif\n\t"
      ".endif\n\t"
      ".if %c[dot]\n\t"
      "vmadot v8, v2, v4, i4\n\t"
      "vmadot v10, v2, v5, i4\n\t"
      "vmadot v12, v2, v6, i4\n\t"
      "vmadot v14, v2, v7, i4\n\t"
      ".endif\n\t"
      ".if !%c[preload_a]\n\t"
      "addi %[a], %[a], 128\n\t"
      ".endif\n\t"
      "addi %[b], %[b], 512\n\t"
      ".endif\n\t"
      ".if !%c[preload_a]\n\t"
      "addi %[a], %[a], 256\n\t"
      ".endif\n\t"
      "addi %[b], %[b], 512\n\t"
      "addi t2, t2, -1\n\t"
      "bnez t2, 2b\n\t"
      ".if %c[mixed]\n\t"
      "j 13f\n\t"
      "12:\n\t"
      ".if %c[load]\n\t"
      "vl1r.v v1, (%[a])\n\t"
      "vl4r.v v4, (%[b])\n\t"
      ".endif\n\t"
      ".if %c[dot]\n\t"
      "vmadot v16, v1, v4, i4\n\t"
      "vmadot v18, v1, v5, i4\n\t"
      "vmadot v20, v1, v6, i4\n\t"
      "vmadot v22, v1, v7, i4\n\t"
      ".endif\n\t"
      "addi %[a], %[a], 128\n\t"
      "addi %[b], %[b], 512\n\t"
      "addi t2, t2, -1\n\t"
      "bnez t2, 12b\n\t"
      "13:\n\t"
      ".endif\n\t"
      ".if %c[scale]\n\t"
      ".if %c[pipe]\n\t"
      "vsetvli t0, zero, e32, m%c[epilogue], ta, ma\n\t"
      ".if %c[mac_combine]\n\t"
      "li t0, 16\n\t"
      "vmacc.vx v8, t0, v16\n\t"
      "vmacc.vx v10, t0, v18\n\t"
      "vmacc.vx v12, t0, v20\n\t"
      "vmacc.vx v14, t0, v22\n\t"
      ".else\n\t"
      "vsll.vi v16, v16, 4\n\t"
      ".if %c[epilogue] == 2\n\t"
      "vsll.vi v18, v18, 4\n\t"
      ".endif\n\t"
      ".if %c[epilogue] <= 4\n\t"
      "vsll.vi v20, v20, 4\n\t"
      ".endif\n\t"
      ".if %c[epilogue] == 2\n\t"
      "vsll.vi v22, v22, 4\n\t"
      ".endif\n\t"
      "vadd.vv v8, v8, v16\n\t"
      ".if %c[epilogue] == 2\n\t"
      "vadd.vv v10, v10, v18\n\t"
      ".endif\n\t"
      ".if %c[epilogue] <= 4\n\t"
      "vadd.vv v12, v12, v20\n\t"
      ".endif\n\t"
      ".if %c[epilogue] == 2\n\t"
      "vadd.vv v14, v14, v22\n\t"
      ".endif\n\t"
      ".endif\n\t"
      ".if %c[early]\n\t"
      ".if !%c[preload_a]\n\t"
      "vl2r.v v0, (%[a])\n\t"
      ".endif\n\t"
      ".if %c[split_a]\n\t"
      "mv t4, %[b]\n\t"
      "bnez t5, 14f\n\t"
      "slli t0, %[sub], 9\n\t"
      "add t4, t4, t0\n\t"
      "14:\n\t"
      ".if %c[f32_scale] == 1\n\t"
      "vl8r.v v16, (t4)\n\t"
      ".else\n\t"
      "vl4r.v v4, (t4)\n\t"
      ".endif\n\t"
      ".else\n\t"
      ".if %c[f32_scale] == 1\n\t"
      "vl8r.v v16, (%[b])\n\t"
      ".else\n\t"
      "vl4r.v v4, (%[b])\n\t"
      ".endif\n\t"
      ".endif\n\t"
      ".endif\n\t"
      "vfcvt.f.x.v v8, v8\n\t"
      ".if %c[epilogue] == 2\n\t"
      "vfcvt.f.x.v v10, v10\n\t"
      ".endif\n\t"
      ".if %c[epilogue] <= 4\n\t"
      "vfcvt.f.x.v v12, v12\n\t"
      ".endif\n\t"
      ".if %c[epilogue] == 2\n\t"
      "vfcvt.f.x.v v14, v14\n\t"
      ".endif\n\t"
      ".else\n\t"
      "vsetvli t0, zero, e32, m8, ta, ma\n\t"
      "vsll.vi v16, v16, 4\n\t"
      "vadd.vv v8, v8, v16\n\t"
      "vfcvt.f.x.v v8, v8\n\t"
      ".endif\n\t"
      ".if !%c[early]\n\t"
      "vl2r.v v0, (%[a])\n\t"
      ".endif\n\t"
      ".if %c[f32_scale]\n\t"
      ".if %c[f32_scale] == 2\n\t"
      "vsetvli t0, zero, e16, m1, ta, ma\n\t"
      "vfwcvt.f.f.v v16, v4\n\t"
      "vfwcvt.f.f.v v18, v5\n\t"
      "vfwcvt.f.f.v v20, v6\n\t"
      "vfwcvt.f.f.v v22, v7\n\t"
      "vsetvli t0, zero, e32, m2, ta, ma\n\t"
      ".endif\n\t"
      "vfmul.vv v16, v16, v0\n\t"
      "vfmul.vv v18, v18, v0\n\t"
      "vfmul.vv v20, v20, v0\n\t"
      "vfmul.vv v22, v22, v0\n\t"
      "vfmacc.vv v24, v8, v16\n\t"
      "vfmacc.vv v26, v10, v18\n\t"
      "vfmacc.vv v28, v12, v20\n\t"
      "vfmacc.vv v30, v14, v22\n\t"
      ".else\n\t"
      ".if %c[epilogue] > 2\n\t"
      "vmv2r.v v2, v0\n\t"
      ".if %c[epilogue] == 8\n\t"
      "vmv4r.v v16, v0\n\t"
      "vmv4r.v v20, v0\n\t"
      ".endif\n\t"
      ".endif\n\t"
      "vsetvli t0, zero, e32, m%c[epilogue], ta, ma\n\t"
      ".if %c[epilogue] == 8\n\t"
      "vfmul.vv v8, v8, v16\n\t"
      ".else\n\t"
      "vfmul.vv v8, v8, v0\n\t"
      ".if %c[epilogue] == 2\n\t"
      "vfmul.vv v10, v10, v0\n\t"
      ".endif\n\t"
      "vfmul.vv v12, v12, v0\n\t"
      ".if %c[epilogue] == 2\n\t"
      "vfmul.vv v14, v14, v0\n\t"
      ".endif\n\t"
      ".endif\n\t"
      ".if %c[pipe]\n\t"
      "vsetvli t0, zero, e16, m%c[narrow], ta, ma\n\t"
      "vfncvt.f.f.w v16, v8\n\t"
      ".if %c[epilogue] == 2\n\t"
      "vfncvt.f.f.w v17, v10\n\t"
      ".endif\n\t"
      ".if %c[epilogue] <= 4\n\t"
      "vfncvt.f.f.w v18, v12\n\t"
      ".endif\n\t"
      ".if %c[epilogue] == 2\n\t"
      "vfncvt.f.f.w v19, v14\n\t"
      ".endif\n\t"
      ".if !%c[early]\n\t"
      "vl4r.v v4, (%[b])\n\t"
      ".endif\n\t"
      ".if %c[matrix_scale]\n\t"
      "vfwmadot v24, v16, v4, fp16\n\t"
      "vfwmadot v26, v17, v5, fp16\n\t"
      "vfwmadot v28, v18, v6, fp16\n\t"
      "vfwmadot v30, v19, v7, fp16\n\t"
      ".else\n\t"
      "vfwmacc.vv v24, v16, v4\n\t"
      ".if %c[epilogue] == 2\n\t"
      "vfwmacc.vv v26, v17, v5\n\t"
      ".endif\n\t"
      ".if %c[epilogue] <= 4\n\t"
      "vfwmacc.vv v28, v18, v6\n\t"
      ".endif\n\t"
      ".if %c[epilogue] == 2\n\t"
      "vfwmacc.vv v30, v19, v7\n\t"
      ".endif\n\t"
      ".endif\n\t"
      ".else\n\t"
      "vsetvli t0, zero, e16, m4, ta, ma\n\t"
      "vfncvt.f.f.w v8, v8\n\t"
      "vl4r.v v4, (%[b])\n\t"
      "vfwmacc.vv v24, v8, v4\n\t"
      ".endif\n\t"
      ".endif\n\t"
      ".endif\n\t"
      "7:\n\t"
      "vsetvli t0, zero, e32, m8, ta, ma\n\t"
      "addi %[a], %[a], 256\n\t"
      ".if %c[split_a]\n\t"
      "xori t5, t5, 1\n\t"
      "bnez t5, 1b\n\t"
      ".endif\n\t"
      "addi %[b], %[b], 512\n\t"
      ".if %c[f32_scale] == 1\n\t"
      "addi %[b], %[b], 512\n\t"
      ".endif\n\t"
      "10:\n\t"
      "addi %[groups], %[groups], -1\n\t"
      ".if %c[mixed]\n\t"
      "xori t3, t3, 1\n\t"
      ".endif\n\t"
      "bnez %[groups], 1b\n\t"
      ".if %c[raw]\n\t"
      "vs8r.v v24, (%[out])\n\t"
      ".else\n\t"
      "vsetvli t0, zero, e32, m1, ta, ma\n\t"
      "vpack.vv v0, v24, v26, 2\n\t"
      "vpack.vv v2, v25, v27, 2\n\t"
      "vpack.vv v4, v28, v30, 2\n\t"
      "vpack.vv v6, v29, v31, 2\n\t"
      "vpack.vv v8, v0, v4, 3\n\t"
      "vpack.vv v10, v1, v5, 3\n\t"
      "vpack.vv v12, v2, v6, 3\n\t"
      "vpack.vv v14, v3, v7, 3\n\t"
      "vse32.v v8, (%[out])\n\t"
      "add t1, %[out], %[stride]\n\t"
      "vse32.v v9, (t1)\n\t"
      "add t1, t1, %[stride]\n\t"
      "vse32.v v10, (t1)\n\t"
      "add t1, t1, %[stride]\n\t"
      "vse32.v v11, (t1)\n\t"
      "add t1, t1, %[stride]\n\t"
      "vse32.v v12, (t1)\n\t"
      "add t1, t1, %[stride]\n\t"
      "vse32.v v13, (t1)\n\t"
      "add t1, t1, %[stride]\n\t"
      "vse32.v v14, (t1)\n\t"
      "add t1, t1, %[stride]\n\t"
      "vse32.v v15, (t1)\n\t"
      ".endif\n\t"
      : [a] "+&r"(a), [b] "+&r"(b), [groups] "+&r"(groups), [active] "+&r"(active)
      : [out] "r"(output), [stride] "r"(stride * sizeof(float)), [sub] "r"(subblocks),
        [dot] "i"((Diagnostic & 7) != 1 && !(Diagnostic & 8)),
        [scale] "i"((Diagnostic & 7) != 2 && !(Diagnostic & 16)),
        [load] "i"((Diagnostic & 7) != 3 && !(Diagnostic & 32)),
        [noload] "i"((Diagnostic & 7) == 3 || (Diagnostic & 32)), [pipe] "i"((Diagnostic & 7) >= 4),
        [early] "i"((Diagnostic & 7) >= 5),
        [epilogue] "i"((Diagnostic & 7) == 6   ? 4
                       : (Diagnostic & 7) == 7 ? 8
                                               : 2),
        [narrow] "i"((Diagnostic & 7) == 6   ? 2
                     : (Diagnostic & 7) == 7 ? 4
                                             : 1),
        [late_coarse] "i"((Diagnostic & 64) != 0), [preload_a] "i"((Diagnostic & 128) != 0),
        [mac_combine] "i"((Diagnostic & 256) != 0), [matrix_scale] "i"((Diagnostic & 512) != 0),
        [f32_scale] "i"((Diagnostic & 1024) ? (Diagnostic & 2048 ? 2 : 1) : 0),
        [mixed] "i"(ResidualA4), [raw] "i"(RawAccumulator), [accumulate] "r"(accumulate),
        [sparse] "i"(Sparse), [radix] "i"(Radix), [split_a] "i"(SplitActivation)
      : "t0", "t1", "t2", "t3", "t4", "t5", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6",
        "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19",
        "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
}

static constexpr int radix_schedule = 3205;

static void tile(const uint8_t *a, const uint8_t *b, int groups, int subblocks, float *output,
                 size_t stride, bool residual_a4 = false, const uint8_t *active = nullptr,
                 bool radix = false, bool split_a = false) {
  if (radix && split_a)
    tile_impl<radix_schedule, false, false, false, true, true>(a, b, groups, subblocks, output,
                                                               stride);
  else if (radix)
    tile_impl<radix_schedule, false, false, false, true>(a, b, groups, subblocks, output, stride);
  else if (active)
    tile_impl<5, true, false, true>(a, b, groups, subblocks, output, stride, false, active);
  else if (residual_a4)
    tile_impl<5, true>(a, b, groups, subblocks, output, stride);
  else
    tile_impl<5>(a, b, groups, subblocks, output, stride);
}

static void tile_accumulate(const uint8_t *a, const uint8_t *b, int groups, int subblocks,
                            float *output, bool accumulate, bool residual_a4,
                            const uint8_t *active = nullptr, bool radix = false,
                            bool split_a = false) {
  if (radix && split_a)
    tile_impl<radix_schedule, false, true, false, true, true>(a, b, groups, subblocks, output, 32,
                                                              accumulate);
  else if (radix)
    tile_impl<radix_schedule, false, true, false, true>(a, b, groups, subblocks, output, 32,
                                                        accumulate);
  else if (active)
    tile_impl<5, true, true, true>(a, b, groups, subblocks, output, 32, accumulate, active);
  else if (residual_a4)
    tile_impl<5, true, true>(a, b, groups, subblocks, output, 32, accumulate);
  else
    tile_impl<5, false, true>(a, b, groups, subblocks, output, 32, accumulate);
}

// The call boundary invalidates compiler assumptions about VL and VTYPE.
__attribute__((noinline)) static void unpack_tile(const float *input, float *output,
                                                  size_t stride) {
  asm volatile("vl8r.v v24, (%[in])\n\t"
               "vsetvli t0, zero, e32, m1, ta, ma\n\t"
               "vpack.vv v0, v24, v26, 2\n\t"
               "vpack.vv v2, v25, v27, 2\n\t"
               "vpack.vv v4, v28, v30, 2\n\t"
               "vpack.vv v6, v29, v31, 2\n\t"
               "vpack.vv v8, v0, v4, 3\n\t"
               "vpack.vv v10, v1, v5, 3\n\t"
               "vpack.vv v12, v2, v6, 3\n\t"
               "vpack.vv v14, v3, v7, 3\n\t"
               "vse32.v v8, (%[out])\n\t"
               "add t1, %[out], %[stride]\n\t"
               "vse32.v v9, (t1)\n\t"
               "add t1, t1, %[stride]\n\t"
               "vse32.v v10, (t1)\n\t"
               "add t1, t1, %[stride]\n\t"
               "vse32.v v11, (t1)\n\t"
               "add t1, t1, %[stride]\n\t"
               "vse32.v v12, (t1)\n\t"
               "add t1, t1, %[stride]\n\t"
               "vse32.v v13, (t1)\n\t"
               "add t1, t1, %[stride]\n\t"
               "vse32.v v14, (t1)\n\t"
               "add t1, t1, %[stride]\n\t"
               "vse32.v v15, (t1)\n\t"
               :
               : [in] "r"(input), [out] "r"(output), [stride] "r"(stride * sizeof(float))
               : "t0", "t1", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",
                 "v10", "v11", "v12", "v13", "v14", "v15", "v24", "v25", "v26", "v27", "v28", "v29",
                 "v30", "v31");
}

} // namespace ime_i4_microkernel

// Same radix16 arithmetic as the frozen M8xN32 kernel. Two M8 rows
// share one N16 weight panel; raw accumulator halves retain native layout.
__attribute__((noinline)) static void radix_m16n16(const uint8_t *a0, const uint8_t *a1,
                                                   const uint8_t *b, int groups, float *out0,
                                                   float *out1, bool accumulate) {
  asm volatile("vsetvli t0, zero, e32, m8, ta, ma\n\t"
               "beqz %[acc], 8f\n\t"
               "vl4r.v v24,(%[o0])\n\t"
               "vl4r.v v28,(%[o1])\n\t"
               "j 9f\n\t"
               "8: vmv.v.i v24,0\n\t"
               "9:\n\t"
               "1:\n\t"
               "vmv.v.i v8,0\n\t"
               "vmv.v.i v16,0\n\t"
               "vsetvli t0,zero,e8,m1,ta,ma\n\t"
               "vl2r.v v0,(%[a0])\n\t"
               "vl2r.v v2,(%[a1])\n\t"
               "li t2,8\n\t"
               "2:\n\t"
               "vl2r.v v4,(%[b])\n\t"
               "addi t3,%[a0],256\n\t"
               "vl1r.v v6,(t3)\n\t"
               "addi t3,%[a1],256\n\t"
               "vl1r.v v7,(t3)\n\t"
               "vmadotus v8,v0,v4,i4\n\t"
               "vmadotus v10,v0,v5,i4\n\t"
               "vmadotus v12,v2,v4,i4\n\t"
               "vmadotus v14,v2,v5,i4\n\t"
               "vmadot v16,v1,v4,i4\n\t"
               "vmadot v18,v1,v5,i4\n\t"
               "vmadot v20,v3,v4,i4\n\t"
               "vmadot v22,v3,v5,i4\n\t"
               "addi t3,%[b],512\n\t"
               "vl2r.v v4,(t3)\n\t"
               "addi %[a0],%[a0],384\n\t"
               "addi %[a1],%[a1],384\n\t"
               "vl2r.v v0,(%[a0])\n\t"
               "vl2r.v v2,(%[a1])\n\t"
               "vmadot v8,v6,v4,i4\n\t"
               "vmadot v10,v6,v5,i4\n\t"
               "vmadot v12,v7,v4,i4\n\t"
               "vmadot v14,v7,v5,i4\n\t"
               "addi %[b],%[b],1024\n\t"
               "addi t2,t2,-1\n\t"
               "bnez t2,2b\n\t"
               "vsetvli t0,zero,e32,m2,ta,ma\n\t"
               "li t0,16\n\t"
               "vmacc.vx v8,t0,v16\n\t"
               "vmacc.vx v10,t0,v18\n\t"
               "vmacc.vx v12,t0,v20\n\t"
               "vmacc.vx v14,t0,v22\n\t"
               "vl2r.v v4,(%[b])\n\t"
               "vfcvt.f.x.v v8,v8\n\t"
               "vfcvt.f.x.v v10,v10\n\t"
               "vfcvt.f.x.v v12,v12\n\t"
               "vfcvt.f.x.v v14,v14\n\t"
               "vsetvli t0,zero,e16,m1,ta,ma\n\t"
               "vfwcvt.f.f.v v16,v4\n\t"
               "vfwcvt.f.f.v v18,v5\n\t"
               "vsetvli t0,zero,e32,m2,ta,ma\n\t"
               "vfmul.vv v20,v16,v2\n\t"
               "vfmul.vv v22,v18,v2\n\t"
               "vfmul.vv v16,v16,v0\n\t"
               "vfmul.vv v18,v18,v0\n\t"
               "vfmacc.vv v24,v8,v16\n\t"
               "vfmacc.vv v26,v10,v18\n\t"
               "vfmacc.vv v28,v12,v20\n\t"
               "vfmacc.vv v30,v14,v22\n\t"
               "addi %[a0],%[a0],256\n\t"
               "addi %[a1],%[a1],256\n\t"
               "addi %[b],%[b],512\n\t"
               "vsetvli t0,zero,e32,m8,ta,ma\n\t"
               "addi %[ng],%[ng],-1\n\t"
               "bnez %[ng],1b\n\t"
               "vs4r.v v24,(%[o0])\n\t"
               "vs4r.v v28,(%[o1])\n\t"
               : [a0] "+&r"(a0), [a1] "+&r"(a1), [b] "+&r"(b), [ng] "+&r"(groups)
               : [o0] "r"(out0), [o1] "r"(out1), [acc] "r"(accumulate)
               : "t0", "t2", "t3", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8",
                 "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20",
                 "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
}
static void radix_m16n32(const uint8_t *a0, const uint8_t *a1, const uint8_t *b, int groups,
                         float *out0, float *out1, bool accumulate) {
  radix_m16n16(a0, a1, b, groups, out0, out1, accumulate);
  radix_m16n16(a0, a1, b + 256, groups, out0 + 128, out1 + 128, accumulate);
}

namespace radix_attention {
using Half = _Float16;
// M<=8 uses four independent accumulators and skips the padded second half.
__attribute__((noinline)) static void ime_tile8(const Half *a, const Half *b, int blocks,
                                                float *tmp) {
  asm volatile("vsetvli t0,zero,e32,m8,ta,ma\n\t"
               "vmv.v.i v16,0\n\t"
               "vsetvli t0,zero,e16,m1,ta,ma\n\t"
               "1:\n\t"
               "vl1re16.v v0,(%[a])\n\t"
               "vl4re16.v v4,(%[b])\n\t"
               "vfwmadot v16,v0,v4,fp16\n\t"
               "vfwmadot v18,v0,v5,fp16\n\t"
               "vfwmadot v20,v0,v6,fp16\n\t"
               "vfwmadot v22,v0,v7,fp16\n\t"
               "addi %[a],%[a],256\n\t"
               "addi %[b],%[b],512\n\t"
               "addi %[blocks],%[blocks],-1\n\t"
               "bnez %[blocks],1b\n\t"
               "vsetvli t0,zero,e32,m1,ta,ma\n\t"
               "vpack.vv v0,v16,v18,2\n\t"
               "vpack.vv v2,v20,v22,2\n\t"
               "vpack.vv v4,v17,v19,2\n\t"
               "vpack.vv v6,v21,v23,2\n\t"
               "vpack.vv v16,v0,v2,3\n\t"
               "vpack.vv v18,v1,v3,3\n\t"
               "vpack.vv v20,v4,v6,3\n\t"
               "vpack.vv v22,v5,v7,3\n\t"
               "vs8r.v v16,(%[tmp])\n\t"
               : [a] "+&r"(a), [b] "+&r"(b), [blocks] "+&r"(blocks)
               : [tmp] "r"(tmp)
               : "t0", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v16", "v17",
                 "v18", "v19", "v20", "v21", "v22", "v23");
}
// Eight independent 8x8 accumulators; full K8 panels. Output is row-major
// 16x32.
__attribute__((noinline)) static void ime_tile(const Half *a, const Half *b, int blocks,
                                               float *tmp) {
  asm volatile("vsetvli t0,zero,e32,m8,ta,ma\n\t"
               "vmv.v.i v16,0\n\t"
               "vmv.v.i v24,0\n\t"
               "vsetvli t0,zero,e16,m1,ta,ma\n\t"
               "vl2re16.v v0,(%[a])\n\t"
               "vl4re16.v v4,(%[b])\n\t"
               "addi %[a],%[a],256\n\t"
               "addi %[b],%[b],512\n\t"
               "1:\n\t"
               "li t0,1\n\t"
               "beq %[blocks],t0,3f\n\t"
               "vl2re16.v v2,(%[a])\n\t"
               "addi %[a],%[a],256\n\t"
               "vfwmadot v16,v0,v4,fp16\n\t"
               "vfwmadot v24,v1,v4,fp16\n\t"
               "vl4re16.v v8,(%[b])\n\t"
               "addi %[b],%[b],512\n\t"
               "vfwmadot v18,v0,v5,fp16\n\t"
               "vfwmadot v26,v1,v5,fp16\n\t"
               "vfwmadot v20,v0,v6,fp16\n\t"
               "vfwmadot v28,v1,v6,fp16\n\t"
               "vfwmadot v22,v0,v7,fp16\n\t"
               "vfwmadot v30,v1,v7,fp16\n\t"
               "addi %[blocks],%[blocks],-1\n\t"
               "beq %[blocks],t0,2f\n\t"
               "vl2re16.v v0,(%[a])\n\t"
               "addi %[a],%[a],256\n\t"
               "vfwmadot v16,v2,v8,fp16\n\t"
               "vfwmadot v24,v3,v8,fp16\n\t"
               "vl4re16.v v4,(%[b])\n\t"
               "addi %[b],%[b],512\n\t"
               "vfwmadot v18,v2,v9,fp16\n\t"
               "vfwmadot v26,v3,v9,fp16\n\t"
               "vfwmadot v20,v2,v10,fp16\n\t"
               "vfwmadot v28,v3,v10,fp16\n\t"
               "vfwmadot v22,v2,v11,fp16\n\t"
               "vfwmadot v30,v3,v11,fp16\n\t"
               "addi %[blocks],%[blocks],-1\n\t"
               "j 1b\n\t"
               "2:\n\t"
               "vfwmadot v16,v2,v8,fp16\n\t"
               "vfwmadot v24,v3,v8,fp16\n\t"
               "vfwmadot v18,v2,v9,fp16\n\t"
               "vfwmadot v26,v3,v9,fp16\n\t"
               "vfwmadot v20,v2,v10,fp16\n\t"
               "vfwmadot v28,v3,v10,fp16\n\t"
               "vfwmadot v22,v2,v11,fp16\n\t"
               "vfwmadot v30,v3,v11,fp16\n\t"
               "j 4f\n\t"
               "3:\n\t"
               "vfwmadot v16,v0,v4,fp16\n\t"
               "vfwmadot v24,v1,v4,fp16\n\t"
               "vfwmadot v18,v0,v5,fp16\n\t"
               "vfwmadot v26,v1,v5,fp16\n\t"
               "vfwmadot v20,v0,v6,fp16\n\t"
               "vfwmadot v28,v1,v6,fp16\n\t"
               "vfwmadot v22,v0,v7,fp16\n\t"
               "vfwmadot v30,v1,v7,fp16\n\t"
               "4:\n\t"
               "vsetvli t0,zero,e32,m1,ta,ma\n\t"
               "vpack.vv v0,v16,v18,2\n\t"
               "vpack.vv v2,v20,v22,2\n\t"
               "vpack.vv v4,v17,v19,2\n\t"
               "vpack.vv v6,v21,v23,2\n\t"
               "vpack.vv v8,v24,v26,2\n\t"
               "vpack.vv v10,v28,v30,2\n\t"
               "vpack.vv v12,v25,v27,2\n\t"
               "vpack.vv v14,v29,v31,2\n\t"
               "vpack.vv v16,v0,v2,3\n\t"
               "vpack.vv v18,v1,v3,3\n\t"
               "vpack.vv v20,v4,v6,3\n\t"
               "vpack.vv v22,v5,v7,3\n\t"
               "vpack.vv v24,v8,v10,3\n\t"
               "vpack.vv v26,v9,v11,3\n\t"
               "vpack.vv v28,v12,v14,3\n\t"
               "vpack.vv v30,v13,v15,3\n\t"
               "vs8r.v v16,(%[tmp])\n\t"
               "addi %[tmp],%[tmp],1024\n\t"
               "vs8r.v v24,(%[tmp])\n\t"
               : [a] "+&r"(a), [b] "+&r"(b), [blocks] "+&r"(blocks), [tmp] "+&r"(tmp)
               :
               : "t0", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
                 "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22",
                 "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
}
__attribute__((noinline)) static void ime_i8_tile(const int8_t *a, const int8_t *b, int blocks,
                                                  int32_t *tmp) {
  asm volatile("vsetvli t0,zero,e32,m8,ta,ma\n\t"
               "vmv.v.i v16,0\n\t"
               "vmv.v.i v24,0\n\t"
               "vsetvli t0,zero,e8,m1,ta,ma\n\t"
               "vl2re8.v v0,(%[a])\n\t"
               "vl4re8.v v4,(%[b])\n\t"
               "addi %[a],%[a],256\n\t"
               "addi %[b],%[b],512\n\t"
               "1:\n\t"
               "li t0,1\n\t"
               "beq %[blocks],t0,3f\n\t"
               "vl2re8.v v2,(%[a])\n\t"
               "addi %[a],%[a],256\n\t"
               "vmadot v16,v0,v4,i8\n\t"
               "vmadot v24,v1,v4,i8\n\t"
               "vl4re8.v v8,(%[b])\n\t"
               "addi %[b],%[b],512\n\t"
               "vmadot v18,v0,v5,i8\n\t"
               "vmadot v26,v1,v5,i8\n\t"
               "vmadot v20,v0,v6,i8\n\t"
               "vmadot v28,v1,v6,i8\n\t"
               "vmadot v22,v0,v7,i8\n\t"
               "vmadot v30,v1,v7,i8\n\t"
               "addi %[blocks],%[blocks],-1\n\t"
               "beq %[blocks],t0,2f\n\t"
               "vl2re8.v v0,(%[a])\n\t"
               "addi %[a],%[a],256\n\t"
               "vmadot v16,v2,v8,i8\n\t"
               "vmadot v24,v3,v8,i8\n\t"
               "vl4re8.v v4,(%[b])\n\t"
               "addi %[b],%[b],512\n\t"
               "vmadot v18,v2,v9,i8\n\t"
               "vmadot v26,v3,v9,i8\n\t"
               "vmadot v20,v2,v10,i8\n\t"
               "vmadot v28,v3,v10,i8\n\t"
               "vmadot v22,v2,v11,i8\n\t"
               "vmadot v30,v3,v11,i8\n\t"
               "addi %[blocks],%[blocks],-1\n\t"
               "j 1b\n\t"
               "2:\n\t"
               "vmadot v16,v2,v8,i8\n\t"
               "vmadot v24,v3,v8,i8\n\t"
               "vmadot v18,v2,v9,i8\n\t"
               "vmadot v26,v3,v9,i8\n\t"
               "vmadot v20,v2,v10,i8\n\t"
               "vmadot v28,v3,v10,i8\n\t"
               "vmadot v22,v2,v11,i8\n\t"
               "vmadot v30,v3,v11,i8\n\t"
               "j 4f\n\t"
               "3:\n\t"
               "vmadot v16,v0,v4,i8\n\t"
               "vmadot v24,v1,v4,i8\n\t"
               "vmadot v18,v0,v5,i8\n\t"
               "vmadot v26,v1,v5,i8\n\t"
               "vmadot v20,v0,v6,i8\n\t"
               "vmadot v28,v1,v6,i8\n\t"
               "vmadot v22,v0,v7,i8\n\t"
               "vmadot v30,v1,v7,i8\n\t"
               "4:\n\t"
               "vsetvli t0,zero,e32,m1,ta,ma\n\t"
               "vpack.vv v0,v16,v18,2\n\t"
               "vpack.vv v2,v20,v22,2\n\t"
               "vpack.vv v4,v17,v19,2\n\t"
               "vpack.vv v6,v21,v23,2\n\t"
               "vpack.vv v8,v24,v26,2\n\t"
               "vpack.vv v10,v28,v30,2\n\t"
               "vpack.vv v12,v25,v27,2\n\t"
               "vpack.vv v14,v29,v31,2\n\t"
               "vpack.vv v16,v0,v2,3\n\t"
               "vpack.vv v18,v1,v3,3\n\t"
               "vpack.vv v20,v4,v6,3\n\t"
               "vpack.vv v22,v5,v7,3\n\t"
               "vpack.vv v24,v8,v10,3\n\t"
               "vpack.vv v26,v9,v11,3\n\t"
               "vpack.vv v28,v12,v14,3\n\t"
               "vpack.vv v30,v13,v15,3\n\t"
               "vs8r.v v16,(%[tmp])\n\t"
               "addi %[tmp],%[tmp],1024\n\t"
               "vs8r.v v24,(%[tmp])\n\t"
               : [a] "+&r"(a), [b] "+&r"(b), [blocks] "+&r"(blocks), [tmp] "+&r"(tmp)
               :
               : "t0", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
                 "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22",
                 "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
}
__attribute__((noinline)) static void ime_tile_raw(const Half *a, const Half *b, int blocks,
                                                   float *tmp, bool accumulate) {
  asm volatile("vsetvli t0,zero,e32,m8,ta,ma\n\t"
               "beqz %[acc],8f\n\t"
               "vl8r.v v16,(%[tmp])\n\t"
               "addi t0,%[tmp],1024\n\t"
               "vl8r.v v24,(t0)\n\t"
               "j 9f\n\t"
               "8: vmv.v.i v16,0\n\t"
               "vmv.v.i v24,0\n\t"
               "9:\n\t"
               "vsetvli t0,zero,e16,m1,ta,ma\n\t"
               "vl2re16.v v0,(%[a])\n\t"
               "vl4re16.v v4,(%[b])\n\t"
               "addi %[a],%[a],256\n\t"
               "addi %[b],%[b],512\n\t"
               "1:\n\t"
               "li t0,1\n\t"
               "beq %[blocks],t0,3f\n\t"
               "vl2re16.v v2,(%[a])\n\t"
               "addi %[a],%[a],256\n\t"
               "vfwmadot v16,v0,v4,fp16\n\t"
               "vfwmadot v24,v1,v4,fp16\n\t"
               "vl4re16.v v8,(%[b])\n\t"
               "addi %[b],%[b],512\n\t"
               "vfwmadot v18,v0,v5,fp16\n\t"
               "vfwmadot v26,v1,v5,fp16\n\t"
               "vfwmadot v20,v0,v6,fp16\n\t"
               "vfwmadot v28,v1,v6,fp16\n\t"
               "vfwmadot v22,v0,v7,fp16\n\t"
               "vfwmadot v30,v1,v7,fp16\n\t"
               "addi %[blocks],%[blocks],-1\n\t"
               "beq %[blocks],t0,2f\n\t"
               "vl2re16.v v0,(%[a])\n\t"
               "addi %[a],%[a],256\n\t"
               "vfwmadot v16,v2,v8,fp16\n\t"
               "vfwmadot v24,v3,v8,fp16\n\t"
               "vl4re16.v v4,(%[b])\n\t"
               "addi %[b],%[b],512\n\t"
               "vfwmadot v18,v2,v9,fp16\n\t"
               "vfwmadot v26,v3,v9,fp16\n\t"
               "vfwmadot v20,v2,v10,fp16\n\t"
               "vfwmadot v28,v3,v10,fp16\n\t"
               "vfwmadot v22,v2,v11,fp16\n\t"
               "vfwmadot v30,v3,v11,fp16\n\t"
               "addi %[blocks],%[blocks],-1\n\t"
               "j 1b\n\t"
               "2:\n\t"
               "vfwmadot v16,v2,v8,fp16\n\t"
               "vfwmadot v24,v3,v8,fp16\n\t"
               "vfwmadot v18,v2,v9,fp16\n\t"
               "vfwmadot v26,v3,v9,fp16\n\t"
               "vfwmadot v20,v2,v10,fp16\n\t"
               "vfwmadot v28,v3,v10,fp16\n\t"
               "vfwmadot v22,v2,v11,fp16\n\t"
               "vfwmadot v30,v3,v11,fp16\n\t"
               "j 4f\n\t"
               "3:\n\t"
               "vfwmadot v16,v0,v4,fp16\n\t"
               "vfwmadot v24,v1,v4,fp16\n\t"
               "vfwmadot v18,v0,v5,fp16\n\t"
               "vfwmadot v26,v1,v5,fp16\n\t"
               "vfwmadot v20,v0,v6,fp16\n\t"
               "vfwmadot v28,v1,v6,fp16\n\t"
               "vfwmadot v22,v0,v7,fp16\n\t"
               "vfwmadot v30,v1,v7,fp16\n\t"
               "4:\n\t"
               "vs8r.v v16,(%[tmp])\n\t"
               "addi %[tmp],%[tmp],1024\n\t"
               "vs8r.v v24,(%[tmp])\n\t"
               : [a] "+&r"(a), [b] "+&r"(b), [blocks] "+&r"(blocks), [tmp] "+&r"(tmp)
               : [acc] "r"(accumulate)
               : "t0", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
                 "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22",
                 "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
}
static vfloat32m2_t expv(vfloat32m2_t x, size_t vl) {
  x = __riscv_vfmax_vf_f32m2(x, -80, vl);
  x = __riscv_vfmin_vf_f32m2(x, 80, vl);
  auto z = __riscv_vfmul_vf_f32m2(x, 1.4426950408889634f, vl);
  auto n = __riscv_vfcvt_x_f_v_i32m2(z, vl);
  auto nf = __riscv_vfcvt_f_x_v_f32m2(n, vl);
  auto r = __riscv_vfsub_vv_f32m2(x, __riscv_vfmul_vf_f32m2(nf, 0.6931471805599453f, vl), vl);
  auto p = __riscv_vfmv_v_f_f32m2(1.f / 720, vl);
  p = __riscv_vfadd_vf_f32m2(__riscv_vfmul_vv_f32m2(p, r, vl), 1.f / 120, vl);
  p = __riscv_vfadd_vf_f32m2(__riscv_vfmul_vv_f32m2(p, r, vl), 1.f / 24, vl);
  p = __riscv_vfadd_vf_f32m2(__riscv_vfmul_vv_f32m2(p, r, vl), 1.f / 6, vl);
  p = __riscv_vfadd_vf_f32m2(__riscv_vfmul_vv_f32m2(p, r, vl), .5f, vl);
  p = __riscv_vfadd_vf_f32m2(__riscv_vfmul_vv_f32m2(p, r, vl), 1.f, vl);
  p = __riscv_vfadd_vf_f32m2(__riscv_vfmul_vv_f32m2(p, r, vl), 1.f, vl);
  auto bits = __riscv_vsll_vx_i32m2(__riscv_vadd_vx_i32m2(n, 127, vl), 23, vl);
  return __riscv_vfmul_vv_f32m2(p, __riscv_vreinterpret_v_i32m2_f32m2(bits), vl);
}

} // namespace radix_attention
using Half = _Float16;
static int round_up(int a, int b) { return (a + b - 1) / b * b; }
static void norm_f32(const float *x, const float *w, int n, float eps, float *y,
                     const float *add = nullptr, float *residual = nullptr) {
  auto acc = __riscv_vfmv_v_f_f32m4(0, 128);
  for (int i = 0; i < n;) {
    size_t vl = __riscv_vsetvl_e32m4(n - i);
    auto v = __riscv_vle32_v_f32m4(x + i, vl);
    if (add)
      v = __riscv_vfadd_vv_f32m4(v, __riscv_vle32_v_f32m4(add + i, vl), vl);
    if (residual)
      __riscv_vse32_v_f32m4(residual + i, v, vl);
    acc = __riscv_vfadd_vv_f32m4_tu(acc, acc, __riscv_vfmul_vv_f32m4(v, v, vl), vl);
    i += vl;
  }
  float sum = __riscv_vfmv_f_s_f32m1_f32(
      __riscv_vfredusum_vs_f32m4_f32m1(acc, __riscv_vfmv_v_f_f32m1(0, 1), 128));
  float inv = 1.f / std::sqrt(sum / n + eps);
  for (int i = 0; i < n;) {
    size_t vl = __riscv_vsetvl_e32m4(n - i);
    auto v = __riscv_vle32_v_f32m4((residual ? residual : x) + i, vl);
    if (add && !residual)
      v = __riscv_vfadd_vv_f32m4(v, __riscv_vle32_v_f32m4(add + i, vl), vl);
    v = __riscv_vfmul_vf_f32m4(v, inv, vl);
    v = __riscv_vfmul_vv_f32m4(v, __riscv_vle32_v_f32m4(w + i, vl), vl);
    __riscv_vse32_v_f32m4(y + i, v, vl);
    i += vl;
  }
}
struct Sync {
  alignas(128) std::atomic<int> a{0};
  alignas(128) std::atomic<int> e{0};
  int nt;
  void wait() {
    if (nt == 1)
      return;
    int old = e.load(std::memory_order_acquire);
    if (a.fetch_add(1, std::memory_order_acq_rel) == nt - 1) {
      a.store(0, std::memory_order_relaxed);
      e.fetch_add(1, std::memory_order_release);
    } else
      while (e.load(std::memory_order_acquire) == old)
        asm volatile("nop");
  }
};
struct Arena {
  uint8_t *p;
  size_t n = 0, cap;
  void *get(size_t b) {
    n = (n + 127) & ~size_t(127);
    auto *q = p + n;
    n += b;
    if (n > cap)
      throw std::runtime_error("TCM budget exceeded");
    return q;
  }
};
// Offline q0/residual merge: u=16*q0+r, signed code=u+8.
// A8*(u+8)-8*sum(A8) reproduces the full A8 residual exactly in INT32.
static void prepare_s8(const float *input, int stride, int k, int rows, uint8_t *dst) {
  if (rows < 8)
    std::memset(dst, 0, size_t(k / 256) * 2560);
  for (int r = 0; r < rows; ++r)
    for (int g = 0; g < k / 256; ++g) {
      auto x = __riscv_vle32_v_f32m8(input + size_t(r) * stride + g * 256, 256);
      auto ax = __riscv_vfabs_v_f32m8(x, 256);
      float mx = __riscv_vfmv_f_s_f32m1_f32(
          __riscv_vfredmax_vs_f32m8_f32m1(ax, __riscv_vfmv_v_f_f32m1(0, 1), 256));
      float scale = mx ? mx / 127 : 1;
      auto q = __riscv_vfcvt_x_f_v_i32m8(__riscv_vfmul_vf_f32m8(x, 1 / scale, 256), 256);
      q = __riscv_vmin_vx_i32m8(__riscv_vmax_vx_i32m8(q, -127, 256), 127, 256);
      int sum = __riscv_vmv_x_s_i32m1_i32(
          __riscv_vredsum_vs_i32m8_i32m1(q, __riscv_vmv_v_x_i32m1(0, 1), 256));
      auto q8 = __riscv_vncvt_x_x_w_i8m2(__riscv_vncvt_x_x_w_i16m4(q, 256), 256);
      auto data = __riscv_vreinterpret_v_i8m2_i64m2(q8);
      auto ix = __riscv_vid_v_u16mf2(32);
      ix = __riscv_vadd_vv_u16mf2(__riscv_vsll_vx_u16mf2(__riscv_vsrl_vx_u16mf2(ix, 1, 32), 7, 32),
                                  __riscv_vsll_vx_u16mf2(__riscv_vand_vx_u16mf2(ix, 1, 32), 3, 32),
                                  32);
      auto *b = dst + size_t(g) * 2560;
      __riscv_vsuxei16_v_i64m2(reinterpret_cast<int64_t *>(b + r * 16), ix, data, 32);
      std::fill_n(reinterpret_cast<int32_t *>(b + 2048) + r * 8, 8, sum * 8);
      std::fill_n(reinterpret_cast<float *>(b + 2304) + r * 8, 8, scale / 2048);
    }
}
__attribute__((noinline)) static void merged_tile_raw(const uint8_t *a, const uint8_t *b,
                                                      int groups, float *out) {
  asm volatile("vsetvli t0,zero,e32,m8,ta,ma\n\t"
               "vmv.v.i v24,0\n\t"
               "1:\n\t"
               "vmv.v.i v8,0\n\t"
               "vmv.v.i v16,0\n\t"
               "vsetvli t0,zero,e8,m1,ta,ma\n\t"
               "li t1,8\n\t"
               "vl1re8.v v0,(%[a])\n\t"
               "2:\n\t"
               "vl4re8.v v4,(%[b])\n\t"
               "vmadot v8,v0,v4,i8\n\t"
               "vmadot v10,v0,v5,i8\n\t"
               "vmadot v12,v0,v6,i8\n\t"
               "vmadot v14,v0,v7,i8\n\t"
               "addi %[a],%[a],128\n\t"
               "addi %[b],%[b],512\n\t"
               "vl1re8.v v0,(%[a])\n\t"
               "vl4re8.v v4,(%[b])\n\t"
               "vmadot v16,v0,v4,i8\n\t"
               "vmadot v18,v0,v5,i8\n\t"
               "vmadot v20,v0,v6,i8\n\t"
               "vmadot v22,v0,v7,i8\n\t"
               "addi %[a],%[a],128\n\t"
               "addi %[b],%[b],512\n\t"
               "vl1re8.v v0,(%[a])\n\t"
               "addi t1,t1,-1\n\t"
               "bnez t1,2b\n\t"
               "vsetvli t0,zero,e32,m8,ta,ma\n\t"
               "vadd.vv v8,v8,v16\n\t"
               "vl2r.v v0,(%[a])\n\t"
               "vsetvli t0,zero,e32,m2,ta,ma\n\t"
               "vsub.vv v8,v8,v0\n\t"
               "vsub.vv v10,v10,v0\n\t"
               "vsub.vv v12,v12,v0\n\t"
               "vsub.vv v14,v14,v0\n\t"
               "vfcvt.f.x.v v8,v8\n\t"
               "vfcvt.f.x.v v10,v10\n\t"
               "vfcvt.f.x.v v12,v12\n\t"
               "vfcvt.f.x.v v14,v14\n\t"
               "addi %[a],%[a],256\n\t"
               "vl2r.v v0,(%[a])\n\t"
               "vl4r.v v4,(%[b])\n\t"
               "vsetvli t0,zero,e16,m1,ta,ma\n\t"
               "vfwcvt.f.f.v v16,v4\n\t"
               "vfwcvt.f.f.v v18,v5\n\t"
               "vfwcvt.f.f.v v20,v6\n\t"
               "vfwcvt.f.f.v v22,v7\n\t"
               "vsetvli t0,zero,e32,m2,ta,ma\n\t"
               "vfmul.vv v16,v16,v0\n\t"
               "vfmul.vv v18,v18,v0\n\t"
               "vfmul.vv v20,v20,v0\n\t"
               "vfmul.vv v22,v22,v0\n\t"
               "vfmacc.vv v24,v8,v16\n\t"
               "vfmacc.vv v26,v10,v18\n\t"
               "vfmacc.vv v28,v12,v20\n\t"
               "vfmacc.vv v30,v14,v22\n\t"
               "addi %[a],%[a],256\n\t"
               "addi %[b],%[b],512\n\t"
               "vsetvli t0,zero,e32,m8,ta,ma\n\t"
               "addi %[g],%[g],-1\n\t"
               "bnez %[g],1b\n\t"
               "vs8r.v v24,(%[out])\n\t"
               : [a] "+&r"(a), [b] "+&r"(b), [g] "+&r"(groups)
               : [out] "r"(out)
               : "memory", "t0", "t1", "v0", "v1", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11",
                 "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
                 "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
}
static void merged_tile(const uint8_t *a, const uint8_t *b, int groups, float *out, int stride) {
  alignas(128) float raw[256];
  merged_tile_raw(a, b, groups, raw);
  ime_i4_microkernel::unpack_tile(raw, out, stride);
}
__attribute__((noinline)) static void merged_m16n16(const uint8_t *a0, const uint8_t *a1,
                                                    const uint8_t *b, int groups, float *out0,
                                                    float *out1, bool accumulate) {
  asm volatile("vsetvli t0, zero, e32, m8, ta, ma\n\t"
               "beqz %[acc], 8f\n\t"
               "vl4r.v v24,(%[o0])\n\t"
               "vl4r.v v28,(%[o1])\n\t"
               "j 9f\n\t"
               "8: vmv.v.i v24,0\n\t"
               "9:\n\t"
               "1:\n\t"
               "vmv.v.i v8,0\n\t"
               "vmv.v.i v16,0\n\t"
               "vsetvli t0,zero,e8,m1,ta,ma\n\t"
               "li t2,16\n\t"
               "2:\n\t"
               "vl1r.v v0,(%[a0])\n\t"
               "vl1r.v v2,(%[a1])\n\t"
               "vl2r.v v4,(%[b])\n\t"
               "vmadot v8,v0,v4,i8\n\t"
               "vmadot v10,v0,v5,i8\n\t"
               "vmadot v12,v2,v4,i8\n\t"
               "vmadot v14,v2,v5,i8\n\t"
               "addi %[a0],%[a0],128\n\t"
               "addi %[a1],%[a1],128\n\t"
               "addi %[b],%[b],512\n\t"
               "addi t2,t2,-1\n\t"
               "bnez t2,2b\n\t"
               "vl2r.v v0,(%[a0])\n\t"
               "vl2r.v v2,(%[a1])\n\t"
               "vsetvli t0,zero,e32,m2,ta,ma\n\t"
               "vsub.vv v8,v8,v0\n\t"
               "vsub.vv v10,v10,v0\n\t"
               "vsub.vv v12,v12,v2\n\t"
               "vsub.vv v14,v14,v2\n\t"
               "addi %[a0],%[a0],256\n\t"
               "addi %[a1],%[a1],256\n\t"
               "vl2r.v v0,(%[a0])\n\t"
               "vl2r.v v2,(%[a1])\n\t"
               "vl2r.v v4,(%[b])\n\t"
               "vfcvt.f.x.v v8,v8\n\t"
               "vfcvt.f.x.v v10,v10\n\t"
               "vfcvt.f.x.v v12,v12\n\t"
               "vfcvt.f.x.v v14,v14\n\t"
               "vsetvli t0,zero,e16,m1,ta,ma\n\t"
               "vfwcvt.f.f.v v16,v4\n\t"
               "vfwcvt.f.f.v v18,v5\n\t"
               "vsetvli t0,zero,e32,m2,ta,ma\n\t"
               "vfmul.vv v20,v16,v2\n\t"
               "vfmul.vv v22,v18,v2\n\t"
               "vfmul.vv v16,v16,v0\n\t"
               "vfmul.vv v18,v18,v0\n\t"
               "vfmacc.vv v24,v8,v16\n\t"
               "vfmacc.vv v26,v10,v18\n\t"
               "vfmacc.vv v28,v12,v20\n\t"
               "vfmacc.vv v30,v14,v22\n\t"
               "addi %[a0],%[a0],256\n\t"
               "addi %[a1],%[a1],256\n\t"
               "addi %[b],%[b],512\n\t"
               "vsetvli t0,zero,e32,m8,ta,ma\n\t"
               "addi %[ng],%[ng],-1\n\t"
               "bnez %[ng],1b\n\t"
               "vs4r.v v24,(%[o0])\n\t"
               "vs4r.v v28,(%[o1])\n\t"
               : [a0] "+&r"(a0), [a1] "+&r"(a1), [b] "+&r"(b), [ng] "+&r"(groups)
               : [o0] "r"(out0), [o1] "r"(out1), [acc] "r"(accumulate)
               : "t0", "t2", "t3", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8",
                 "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20",
                 "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
}
static void merged_m16n32(const uint8_t *a, const uint8_t *b, const uint8_t *w, int groups,
                          float *x, float *y) {
  merged_m16n16(a, b, w, groups, x, y, false);
  merged_m16n16(a, b, w + 256, groups, x + 128, y + 128, false);
}
enum class WeightType { NVFP4, BF16, PackedG256, PackedFP16, PackedS8, PackedFullI8 };
struct Matrix {
  int n = 0, k = 0;
  WeightType type = WeightType::NVFP4;
  const uint8_t *raw = nullptr;
  size_t bytes = 0;
  float factor = 256.f;
  const float *scales = nullptr;
  void validate() const {
    const size_t expected = type == WeightType::PackedFullI8 ? size_t((n + 31) / 32) * 32 * k
                            : type == WeightType::NVFP4      ? size_t(n) * k / 64 * 36
                            : (type == WeightType::PackedG256 || type == WeightType::PackedS8)
                                ? size_t((n + 31) / 32) * (k / 256) * 8704
                            : type == WeightType::PackedFP16 ? size_t((n + 31) / 32) * 32 * k * 2
                                                             : size_t(n) * k * 2;
    if (n <= 0 || k <= 0 || k % 256 || !raw || bytes != expected || !std::isfinite(factor) ||
        factor <= 0)
      throw std::invalid_argument("matrix format/shape/length");
  }
};

// Decode up to 128 contiguous K values, starting on a K64 boundary.
// No DDR expansion or persistent alternative weight copy.
static inline vfloat32m4_t nv_decode(const uint8_t *src, size_t vl) {
  auto i = __riscv_vid_v_u8m1(vl);
  auto block = __riscv_vmul_vx_u8m1(__riscv_vsrl_vx_u8m1(i, 6, vl), 36, vl);
  auto group = __riscv_vsrl_vx_u8m1(__riscv_vand_vx_u8m1(i, 63, vl), 4, vl);
  auto ix = __riscv_vadd_vv_u8m1(
      block,
      __riscv_vadd_vv_u8m1(__riscv_vmul_vx_u8m1(group, 8, vl), __riscv_vand_vx_u8m1(i, 7, vl), vl),
      vl);
  ix = __riscv_vadd_vx_u8m1(ix, 4, vl);
  auto raw = __riscv_vle8_v_u8m1(src, ((vl + 63) / 64) * 36);
  auto p = __riscv_vrgather_vv_u8m1(raw, ix, vl);
  auto sh =
      __riscv_vsll_vx_u8m1(__riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(i, 3, vl), 1, vl), 2, vl);
  auto q = __riscv_vand_vx_u8m1(__riscv_vsrl_vv_u8m1(p, sh, vl), 15, vl);
  auto sc = __riscv_vrgather_vv_u8m1(raw, __riscv_vadd_vv_u8m1(block, group, vl), vl);
  auto su = __riscv_vzext_vf4_u32m4(sc, vl);
  auto bits = __riscv_vsll_vx_u32m4(__riscv_vadd_vx_u32m4(su, 960, vl), 20, vl);
  auto scale = __riscv_vreinterpret_v_u32m4_f32m4(bits);
  auto sub = __riscv_vfmul_vf_f32m4(__riscv_vfcvt_f_xu_v_f32m4(su, vl), 0x1p-9f, vl);
  scale = __riscv_vmerge_vvm_f32m4(scale, sub, __riscv_vmsltu_vx_u32m4_b8(su, 8, vl), vl);
  scale = __riscv_vfmerge_vfm_f32m4(scale, 0, __riscv_vmseq_vx_u32m4_b8(su, 127, vl), vl);
  const float lut[8] = {0, .5f, 1, 1.5f, 2, 3, 4, 6};
  auto table = __riscv_vle32_v_f32m4(lut, 8);
  auto index = __riscv_vzext_vf4_u32m4(__riscv_vand_vx_u8m1(q, 7, vl), vl);
  auto value = __riscv_vrgather_vv_f32m4(table, index, vl);
  auto sign =
      __riscv_vsll_vx_u32m4(__riscv_vzext_vf4_u32m4(__riscv_vand_vx_u8m1(q, 8, vl), vl), 28, vl);
  value = __riscv_vreinterpret_v_u32m4_f32m4(
      __riscv_vxor_vv_u32m4(__riscv_vreinterpret_v_f32m4_u32m4(value), sign, vl));
  return __riscv_vfmul_vv_f32m4(value, scale, vl);
}
static inline vfloat32m4_t bf_decode(const uint16_t *src, size_t vl) {
  auto b = __riscv_vle16_v_u16m2(src, vl);
  return __riscv_vreinterpret_v_u32m4_f32m4(
      __riscv_vsll_vx_u32m4(__riscv_vzext_vf2_u32m4(b, vl), 16, vl));
}
static void weight16(const Matrix &w, int col, int first, int count, Half *dst) {
  if (w.type == WeightType::PackedFP16) {
    fp4_residual::copy_gu_input(dst, w.raw + (size_t(col / 32) * w.k + first) * 64,
                                size_t(count) * 64);
    return;
  }
  for (int n = 0; n < 32; ++n) {
    if (col + n >= w.n) {
      for (int z = 0; z < count; z += 8)
        std::fill_n(dst + size_t(z / 8) * 256 + n * 8, 8, Half(0));
      continue;
    }
    for (int z = 0; z < count; z += 128) {
      const size_t vl = std::min(128, count - z);
      auto x = w.type == WeightType::NVFP4
                   ? nv_decode(w.raw + (size_t(col + n) * w.k + first + z) / 64 * 36, vl)
                   : bf_decode(reinterpret_cast<const uint16_t *>(w.raw) + size_t(col + n) * w.k +
                                   first + z,
                               vl);
      x = __riscv_vfmul_vf_f32m4(x, w.factor, vl);
      auto half = __riscv_vfncvt_f_f_w_f16m2(x, vl);
      auto words = __riscv_vreinterpret_v_u16m2_u64m2(__riscv_vreinterpret_v_f16m2_u16m2(half));
      const size_t nw = vl / 4;
      auto i = __riscv_vid_v_u16mf2(nw);
      auto index = __riscv_vadd_vv_u16mf2(
          __riscv_vsll_vx_u16mf2(__riscv_vsrl_vx_u16mf2(i, 1, nw), 9, nw),
          __riscv_vsll_vx_u16mf2(__riscv_vand_vx_u16mf2(i, 1, nw), 3, nw), nw);
      __riscv_vsuxei16_v_u64m2(reinterpret_cast<uint64_t *>(dst + size_t(z / 8) * 256 + n * 8),
                               index, words, nw);
    }
  }
}
static void nibble_store(vint32m8_t value, uint8_t *dst) {
  auto q = __riscv_vreinterpret_v_i8m2_u8m2(
      __riscv_vncvt_x_x_w_i8m2(__riscv_vncvt_x_x_w_i16m4(value, 256), 256));
  auto pair = __riscv_vreinterpret_v_u8m2_u16m2(q);
  auto p = __riscv_vor_vv_u16m2(
      __riscv_vand_vx_u16m2(pair, 15, 128),
      __riscv_vand_vx_u16m2(__riscv_vsrl_vx_u16m2(pair, 4, 128), 240, 128), 128);
  auto bytes = __riscv_vncvt_x_x_w_u8m1(p, 128);
  auto words = __riscv_vreinterpret_v_u8m1_u64m1(bytes);
  auto i = __riscv_vid_v_u16mf4(16);
  auto index =
      __riscv_vadd_vv_u16mf4(__riscv_vsll_vx_u16mf4(__riscv_vsrl_vx_u16mf4(i, 1, 16), 10, 16),
                             __riscv_vsll_vx_u16mf4(__riscv_vand_vx_u16mf4(i, 1, 16), 3, 16), 16);
  __riscv_vsuxei16_v_u64m1(reinterpret_cast<uint64_t *>(dst), index, words, 16);
}
static void weight_i4(const Matrix &w, int col, uint8_t *dst) {
  if (w.type == WeightType::PackedG256 || w.type == WeightType::PackedS8) {
    const size_t stride = size_t(w.k / 256) * 8704;
    fp4_residual::copy_gu_input(dst, w.raw + size_t(col / 32) * stride, stride);
    return;
  }
  alignas(128) float values[256];
  for (int g = 0; g < w.k / 256; ++g)
    for (int n = 0; n < 32; ++n) {
      auto *b = dst + size_t(g) * 8704;
      if (col + n >= w.n) {
        for (int z = 0; z < 8; ++z) {
          std::memset(b + z * 1024 + n * 16, 0, 16);
          std::memset(b + z * 1024 + 512 + n * 16, 0, 16);
        }
        for (int r = 0; r < 8; ++r)
          reinterpret_cast<Half *>(b + 8192)[(n / 8) * 64 + r * 8 + n % 8] = Half(2048);
        continue;
      }
      const auto *src = w.raw + (size_t(col + n) * w.k + g * 256) / 64 * 36;
      __riscv_vse32_v_f32m4(values, nv_decode(src, 128), 128);
      __riscv_vse32_v_f32m4(values + 128, nv_decode(src + 72, 128), 128);
      auto v = __riscv_vle32_v_f32m8(values, 256);
      auto zero = __riscv_vfmv_v_f_f32m1(0, 1);
      float mx = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmax_vs_f32m8_f32m1(v, zero, 256));
      float mn = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmin_vs_f32m8_f32m1(v, zero, 256));
      float s = std::max(mx / 7.4375f, -mn / 8.5f);
      if (!(s > 0))
        s = 1;
      const Half hs = Half(s * 2048);
      s = float(hs) / 2048;
      auto u = __riscv_vfcvt_x_f_v_i32m8(__riscv_vfmul_vf_f32m8(v, 16 / s, 256), 256);
      u = __riscv_vmin_vx_i32m8(__riscv_vmax_vx_i32m8(u, -136, 256), 119, 256);
      auto main = __riscv_vsra_vx_i32m8(__riscv_vadd_vx_i32m8(u, 8, 256), 4, 256);
      auto residual = __riscv_vsub_vv_i32m8(u, __riscv_vsll_vx_i32m8(main, 4, 256), 256);
      nibble_store(main, b + n * 16);
      nibble_store(residual, b + 512 + n * 16);
      for (int r = 0; r < 8; ++r)
        reinterpret_cast<Half *>(b + 8192)[(n / 8) * 64 + r * 8 + n % 8] = hs;
    }
}
static void pack16_rows(const float *src, int stride, int rows, int k, Half *dst) {
  std::fill_n(dst, size_t(16) * k, Half(0));
  for (int r = 0; r < rows; ++r)
    for (int z = 0; z < k; z += 128) {
      const size_t vl = std::min(128, k - z);
      auto x = __riscv_vle32_v_f32m4(src + size_t(r) * stride + z, vl);
      auto y = __riscv_vfncvt_f_f_w_f16m2(x, vl);
      auto words = __riscv_vreinterpret_v_u16m2_u64m2(__riscv_vreinterpret_v_f16m2_u16m2(y));
      const size_t nw = vl / 4;
      auto i = __riscv_vid_v_u16mf2(nw);
      auto index = __riscv_vadd_vv_u16mf2(
          __riscv_vsll_vx_u16mf2(__riscv_vsrl_vx_u16mf2(i, 1, nw), 8, nw),
          __riscv_vsll_vx_u16mf2(__riscv_vand_vx_u16mf2(i, 1, nw), 3, nw), nw);
      __riscv_vsuxei16_v_u64m2(reinterpret_cast<uint64_t *>(dst + size_t(z / 8) * 128 + r * 8),
                               index, words, nw);
    }
}
// Packed A is [ceil(M/16), K/8, 16, 8]. B remains BF16 or NVFP4 in DDR.
// The full M slice reuses each converted K/N panel inside real TCM.
static void linear16(const Matrix &w, const Half *ap, int m, int first_row, int col, int width,
                     float *out, int ldc, uint8_t *scratch, size_t bytes) {
  const float inverse = 1.f / w.factor;
  const int mp = round_up(m, 16);
  const int kc = std::min(w.k, int((bytes - 2048) / (2 * (mp + 32))) / 64 * 64);
  if (kc < 64)
    throw std::runtime_error("FP16/BF16 linear TCM budget");
  Arena arena{scratch, 0, bytes};
  auto *a = (Half *)arena.get(size_t(mp) * kc * 2);
  auto *b = (Half *)arena.get(size_t(kc) * 32 * 2);
  auto *tmp = (float *)arena.get(2048);
  for (int k0 = 0; k0 < w.k; k0 += kc) {
    const int count = std::min(kc, w.k - k0);
    for (int r = 0; r < mp; r += 16)
      fp4_residual::copy_gu_input(a + size_t(r) * count,
                                  ap + (size_t((first_row + r) / 16) * w.k * 16 + size_t(k0) * 16),
                                  size_t(count) * 16 * 2);
    for (int n = 0; n < width; n += 32) {
      weight16(w, col + n, k0, count, b);
      for (int r = 0; r < m; r += 16) {
        radix_attention::ime_tile(a + size_t(r) * count, b, count / 8, tmp);
        for (int rr = 0; rr < std::min(16, m - r); ++rr) {
          const int valid = std::min(32, width - n);
          auto v =
              __riscv_vfmul_vf_f32m1(__riscv_vle32_v_f32m1(tmp + rr * 32, valid), inverse, valid);
          auto *dst = out + size_t(r + rr) * ldc + n;
          if (k0)
            v = __riscv_vfadd_vv_f32m1(v, __riscv_vle32_v_f32m1(dst, valid), valid);
          __riscv_vse32_v_f32m1(dst, v, valid);
        }
      }
    }
  }
}
static void linear_i4(const Matrix &w, const uint8_t *ap, int m, int first_row, int col, int width,
                      float *out, int ldc, uint8_t *scratch, size_t bytes) {
  const bool merged = w.type == WeightType::PackedS8;
  const size_t astride = size_t(w.k / 256) * (merged ? 2560 : 3328),
               bstride = size_t(w.k / 256) * 8704;
  if (bytes <= bstride + 1024 + astride)
    throw std::runtime_error("radix linear TCM budget");
  const int panels = std::max(1, std::min((m + 7) / 8, int((bytes - bstride - 1024) / astride)));
  Arena arena{scratch, 0, bytes};
  auto *b = (uint8_t *)arena.get(bstride);
  auto *a = (uint8_t *)arena.get(size_t(panels) * astride);
  auto *tmp = (float *)arena.get(1024);
  for (int n = 0; n < width; n += 32) {
    weight_i4(w, col + n, b);
    for (int r = 0; r < m; r += panels * 8) {
      const int count = std::min(panels, (m - r + 7) / 8);
      a = const_cast<uint8_t *>(ap + size_t((first_row + r) / 8) * astride);
      for (int p = 0; p < count; ++p) {
        if (merged) {
          merged_tile(a + size_t(p) * astride, b, w.k / 256, tmp, 32);
          for (int rr = 0; rr < std::min(8, m - r - p * 8); ++rr)
            std::memcpy(out + size_t(r + p * 8 + rr) * ldc + n, tmp + rr * 32,
                        size_t(std::min(32, width - n)) * 4);
        } else if (m - r - p * 8 >= 8 && width - n >= 32)
          ime_i4_microkernel::tile(a + size_t(p) * astride, b, w.k / 256, 8,
                                   out + size_t(r + p * 8) * ldc + n, ldc, false, nullptr, true);
        else {
          ime_i4_microkernel::tile(a + size_t(p) * astride, b, w.k / 256, 8, tmp, 32, false,
                                   nullptr, true);
          for (int rr = 0; rr < std::min(8, m - r - p * 8); ++rr)
            std::memcpy(out + size_t(r + p * 8 + rr) * ldc + n, tmp + rr * 32,
                        size_t(std::min(32, width - n)) * 4);
        }
      }
    }
  }
}

// BF16 building blocks: BF16 storage, FP32 math for conversion/normalization.
// GEMM below uses an explicitly scaled FP16 IME bridge, not native BF16 IME.
void bf16_to_fp32(const uint16_t *x, float *y, size_t n) {
  for (size_t i = 0; i < n;) {
    size_t vl = __riscv_vsetvl_e32m4(n - i);
    __riscv_vse32_v_f32m4(y + i, bf_decode(x + i, vl), vl);
    i += vl;
  }
}
void fp32_to_bf16(const float *x, uint16_t *y, size_t n) {
  for (size_t i = 0; i < n;) {
    size_t vl = __riscv_vsetvl_e32m4(n - i);
    auto b = __riscv_vreinterpret_v_f32m4_u32m4(__riscv_vle32_v_f32m4(x + i, vl));
    auto odd = __riscv_vand_vx_u32m4(__riscv_vsrl_vx_u32m4(b, 16, vl), 1, vl);
    auto r = __riscv_vadd_vv_u32m4(__riscv_vadd_vx_u32m4(b, 0x7fff, vl), odd, vl);
    auto nan = __riscv_vmsgtu_vx_u32m4_b8(__riscv_vand_vx_u32m4(b, 0x7fffffff, vl), 0x7f800000, vl);
    r = __riscv_vmerge_vvm_u32m4(r, __riscv_vor_vx_u32m4(b, 0x400000, vl), nan, vl);
    auto v = __riscv_vncvt_x_x_w_u16m2(__riscv_vsrl_vx_u32m4(r, 16, vl), vl);
    __riscv_vse16_v_u16m2(y + i, v, vl);
    i += vl;
  }
}
void bf16_rmsnorm(const uint16_t *x, const float *weight, int rows, int features, float epsilon,
                  float *y) {
  if (!x || !weight || !y || rows < 1 || features < 1 || !(epsilon > 0))
    throw std::invalid_argument("BF16 RMSNorm arguments");
  for (int r = 0; r < rows; ++r) {
    auto *out = y + size_t(r) * features;
    bf16_to_fp32(x + size_t(r) * features, out, features);
    norm_f32(out, weight, features, epsilon, out);
  }
}
// Caller owns workspace and TCM; no persistent allocations or thread creation.
// A is row-major BF16, B is [N,K] BF16 or offline PackedFP16, C is FP32 [M,N].
// This bridge admits finite values whose scaled FP16 approximation is intended.
// It does not promise the full BF16 exponent range without scaling.
void bf16_gemm_fp16_bridge(const uint16_t *a, int rows, int stride, const Matrix &b, float *out,
                           int ldc, Half *ap, float *row_scratch, uint8_t *tcm, size_t tcm_bytes,
                           float activation_factor = 1) {
  b.validate();
  if (!a || !out || !ap || !row_scratch || !tcm || rows < 1 || stride < b.k || ldc < b.n ||
      (b.type != WeightType::BF16 && b.type != WeightType::PackedFP16) ||
      !(activation_factor > 0) || !std::isfinite(activation_factor))
    throw std::invalid_argument("BF16 GEMM bridge arguments");
  for (int first = 0; first < rows; first += 16) {
    int count = std::min(16, rows - first);
    for (int r = 0; r < count; ++r) {
      auto *dst = row_scratch + size_t(r) * b.k;
      bf16_to_fp32(a + size_t(first + r) * stride, dst, b.k);
      for (int c = 0; c < b.k;) {
        size_t vl = __riscv_vsetvl_e32m4(b.k - c);
        auto x = __riscv_vfmul_vf_f32m4(__riscv_vle32_v_f32m4(dst + c, vl), activation_factor, vl);
        __riscv_vse32_v_f32m4(dst + c, x, vl);
        c += vl;
      }
    }
    pack16_rows(row_scratch, b.k, count, b.k, ap + size_t(first) * b.k);
  }
  for (int first = 0; first < rows; first += 128)
    linear16(b, ap, std::min(128, rows - first), first, 0, b.n, out + size_t(first) * ldc, ldc, tcm,
             tcm_bytes);
  if (activation_factor != 1)
    for (int r = 0; r < rows; ++r)
      for (int c = 0; c < b.n;) {
        size_t vl = __riscv_vsetvl_e32m4(b.n - c);
        auto *p = out + size_t(r) * ldc + c;
        __riscv_vse32_v_f32m4(
            p, __riscv_vfdiv_vf_f32m4(__riscv_vle32_v_f32m4(p, vl), activation_factor, vl), vl);
        c += vl;
      }
}

// Native accumulator layout survives K tiles; unpack and rescale once.
static void linear16_raw(const Matrix &w, const Half *ap, int rows, int first, int col, int width,
                         float *raw, uint8_t *scratch, size_t bytes) {
  int mp = round_up(rows, 16);
  int kc = std::min(w.k, int(bytes / (2 * (mp + 32))) / 64 * 64);
  if (kc < 64)
    throw std::runtime_error("linear16_raw budget");
  auto *a = (Half *)scratch;
  auto *b = a + size_t(mp) * kc;
  for (int k0 = 0; k0 < w.k; k0 += kc) {
    int count = std::min(kc, w.k - k0);
    for (int r = 0; r < mp; r += 16)
      fp4_residual::copy_gu_input(a + size_t(r) * count,
                                  ap + size_t((first + r) / 16) * w.k * 16 + size_t(k0) * 16,
                                  size_t(count) * 32);
    for (int n = 0; n < width; n += 32) {
      weight16(w, col + n, k0, count, b);
      for (int r = 0; r < rows; r += 16)
        radix_attention::ime_tile_raw(a + size_t(r) * count, b, count / 8,
                                      raw + size_t(n / 32) * mp * 32 + size_t(r) * 32, k0 != 0);
    }
  }
}

static void pack_full_i8(const float *src, int stride, int rows, int k, uint8_t *dst, float *scales,
                         const float *smooth) {
  std::memset(dst, 0, size_t(16) * k);
  alignas(128) float line[8192];
  for (int r = 0; r < rows; ++r) {
    auto mx = __riscv_vfmv_v_f_f32m4(0, 128);
    for (int c = 0; c < k;) {
      size_t vl = __riscv_vsetvl_e32m4(k - c);
      auto x = __riscv_vle32_v_f32m4(src + size_t(r) * stride + c, vl);
      if (smooth)
        x = __riscv_vfmul_vv_f32m4(x, __riscv_vle32_v_f32m4(smooth + c, vl), vl);
      __riscv_vse32_v_f32m4(line + c, x, vl);
      mx = __riscv_vfmax_vv_f32m4_tu(mx, mx, __riscv_vfabs_v_f32m4(x, vl), vl);
      c += vl;
    }
    float maximum = __riscv_vfmv_f_s_f32m1_f32(
        __riscv_vfredmax_vs_f32m4_f32m1(mx, __riscv_vfmv_v_f_f32m1(0, 1), 128));
    float scale = maximum ? maximum / 127 : 1;
    scales[r] = scale;
    for (int c = 0; c < k; c += 128) {
      size_t vl = std::min(128, k - c);
      auto x = __riscv_vfmul_vf_f32m4(__riscv_vle32_v_f32m4(line + c, vl), 1 / scale, vl);
      auto q = __riscv_vfcvt_x_f_v_i32m4(x, vl);
      q = __riscv_vmin_vx_i32m4(__riscv_vmax_vx_i32m4(q, -127, vl), 127, vl);
      auto bytes = __riscv_vncvt_x_x_w_i8m1(__riscv_vncvt_x_x_w_i16m2(q, vl), vl);
      auto words = __riscv_vreinterpret_v_i8m1_i64m1(bytes);
      size_t nw = vl / 8;
      auto ix = __riscv_vid_v_u16mf4(nw);
      ix = __riscv_vadd_vv_u16mf4(__riscv_vsll_vx_u16mf4(__riscv_vsrl_vx_u16mf4(ix, 1, nw), 8, nw),
                                  __riscv_vsll_vx_u16mf4(__riscv_vand_vx_u16mf4(ix, 1, nw), 3, nw),
                                  nw);
      __riscv_vsuxei16_v_i64m1(reinterpret_cast<int64_t *>(dst + size_t(c / 16) * 256 + r * 16), ix,
                               words, nw);
    }
  }
}
static void linear_full_i8(const Matrix &w, const uint8_t *ap, const float *as, int rows, int first,
                           int col, int width, float *out, int ldc, uint8_t *scratch,
                           size_t bytes) {
  int mp = round_up(rows, 16), kc = std::min(w.k, int(bytes / (mp + 32)) / 64 * 64);
  if (kc < 64 || !w.scales)
    throw std::runtime_error("full INT8 shape/scales");
  auto *a = reinterpret_cast<int8_t *>(scratch);
  auto *b = a + size_t(mp) * kc;
  alignas(128) int32_t tmp[512];
  for (int k0 = 0; k0 < w.k; k0 += kc) {
    int count = std::min(kc, w.k - k0);
    bool last = k0 + count == w.k;
    for (int r = 0; r < mp; r += 16)
      fp4_residual::copy_gu_input(a + size_t(r) * count,
                                  ap + size_t((first + r) / 16) * w.k * 16 + size_t(k0) * 16,
                                  size_t(count) * 16);
    for (int n = 0; n < width; n += 32) {
      fp4_residual::copy_gu_input(b, w.raw + (size_t((col + n) / 32) * w.k + k0) * 32,
                                  size_t(count) * 32);
      for (int r = 0; r < rows; r += 16) {
        radix_attention::ime_i8_tile(a + size_t(r) * count, b, count / 16, tmp);
        for (int rr = 0; rr < std::min(16, rows - r); ++rr) {
          size_t vl = std::min(32, width - n);
          auto *dst = out + size_t(r + rr) * ldc + n;
          auto q = __riscv_vle32_v_i32m1(tmp + rr * 32, vl);
          if (k0)
            q = __riscv_vadd_vv_i32m1(
                q, __riscv_vreinterpret_v_f32m1_i32m1(__riscv_vle32_v_f32m1(dst, vl)), vl);
          if (last) {
            auto sc = __riscv_vfmul_vf_f32m1(__riscv_vle32_v_f32m1(w.scales + col + n, vl),
                                             as[first + r + rr], vl);
            __riscv_vse32_v_f32m1(
                dst, __riscv_vfmul_vv_f32m1(__riscv_vfcvt_f_x_v_f32m1(q, vl), sc, vl), vl);
          } else
            __riscv_vse32_v_f32m1(dst, __riscv_vreinterpret_v_i32m1_f32m1(q), vl);
        }
      }
    }
  }
}
// Full BF16 exponent range fallback. No FP16 conversion, arbitrary positive K/N.
void bf16_gemm_fp32(const uint16_t *a, int rows, int lda, const Matrix &b, float *out, int ldc) {
  if (!a || !out || !b.raw || rows < 1 || b.n < 1 || b.k < 1 || lda < b.k || ldc < b.n ||
      b.type != WeightType::BF16 || b.bytes != size_t(b.n) * b.k * 2)
    throw std::invalid_argument("BF16 FP32 GEMM arguments");
  auto *w = reinterpret_cast<const uint16_t *>(b.raw);
  for (int r = 0; r < rows; ++r)
    for (int n = 0; n < b.n; ++n) {
      auto acc = __riscv_vfmv_v_f_f32m4(0, 128);
      for (int k = 0; k < b.k;) {
        size_t vl = __riscv_vsetvl_e32m4(b.k - k);
        auto x = bf_decode(a + size_t(r) * lda + k, vl), y = bf_decode(w + size_t(n) * b.k + k, vl);
        acc = __riscv_vfadd_vv_f32m4_tu(acc, acc, __riscv_vfmul_vv_f32m4(x, y, vl), vl);
        k += vl;
      }
      out[size_t(r) * ldc + n] = __riscv_vfmv_f_s_f32m1_f32(
          __riscv_vfredusum_vs_f32m4_f32m1(acc, __riscv_vfmv_v_f_f32m1(0, 1), 128));
    }
}
// Preconverted PackedFP16 selects the admitted IME bridge. Raw BF16 selects
// FP32 RVV, preserving BF16 range rather than silently overflowing FP16.
void bf16_gemm(const uint16_t *a, int rows, int lda, const Matrix &b, float *out, int ldc,
               Half *ap = nullptr, float *row_scratch = nullptr, uint8_t *tcm = nullptr,
               size_t tcm_bytes = 0, float activation_factor = 1) {
  if (b.type == WeightType::BF16)
    bf16_gemm_fp32(a, rows, lda, b, out, ldc);
  else if (b.type == WeightType::PackedFP16)
    bf16_gemm_fp16_bridge(a, rows, lda, b, out, ldc, ap, row_scratch, tcm, tcm_bytes,
                          activation_factor);
  else
    throw std::invalid_argument("BF16 GEMM storage type");
}
struct BlockParameters {
  Matrix q, k, v, o, gate, up, down;
  const float *input_smooth = nullptr, *o_smooth = nullptr, *ffn_smooth = nullptr,
              *down_smooth = nullptr;
  const float *attn_norm = nullptr, *q_norm = nullptr, *k_norm = nullptr, *ffn_norm = nullptr;
  int head_dim = 128;
  float epsilon = 1e-6f, rope_base = 5000000.f;
};
struct DebugStages {
  std::vector<float> norm_attn, q_norm, k_norm, q_rope, k_rope, v_heads, probabilities,
      attention_context, attention_out, residual_attn, norm_ffn, gate, up, swiglu, down, output;
};

class PrefillBlock {
  BlockParameters p;
  int m, mp, sp, nt, e, f, h, hk;
  int capacity;
  bool borrowed_tcm, mrope = false;
  bool radix, merged, full;
  std::vector<float> ascale;
  int ast;
  AlignedVector<uint8_t> ap;
  AlignedVector<Half> op, zp;
  AlignedVector<float> swiglu_full;
  std::vector<float> residual, cs, sn;
  std::vector<uint8_t *> tcm;
  std::vector<std::thread> workers;
  pthread_barrier_t command;
  Sync sync;
  std::atomic<bool> stop{false};
  std::atomic<int> ready{0};
  const Half *input = nullptr;
  Half *output = nullptr;
  const int64_t *positions = nullptr;
  const float *mask = nullptr;
  DebugStages *debug = nullptr;
  static constexpr size_t TCM = 393216;
  void release() {
    for (int t = 0; t < int(tcm.size()); ++t)
      if (tcm[t]) {
        if (!borrowed_tcm) spine_tcm_mem_release(t);
        spine_tcm_mem_free(t);
        tcm[t] = nullptr;
      }
  }
  void project(const Matrix &w, int first, int rows, int col, int width, float *dst, int stride,
               int t, size_t occupied) {
    if (full)
      linear_full_i8(w, ap.data(), ascale.data(), rows, first, col, width, dst, stride,
                     tcm[t] + occupied, TCM - occupied);
    else if (radix)
      linear_i4(w, ap.data(), rows, first, col, width, dst, stride, tcm[t] + occupied,
                TCM - occupied);
    else
      linear16(w, reinterpret_cast<const Half *>(ap.data()), rows, first, col, width, dst, stride,
               tcm[t] + occupied, TCM - occupied);
  }
  void prepare(int t, bool after) {
    const int batch = radix && !full ? 8 : 16;
    auto *tmp = reinterpret_cast<float *>(tcm[t]);
    for (int first = t * batch; first < m; first += nt * batch) {
      const int rows = std::min(batch, m - first);
      for (int r = 0; r < rows; ++r) {
        const int row = first + r;
        if (after)
          norm_f32(residual.data() + size_t(row) * e, p.ffn_norm, e, p.epsilon, tmp + r * e);
        else {
          for (int j = 0; j < e;) {
            size_t vl = __riscv_vsetvl_e16m2(e - j);
            __riscv_vse32_v_f32m4(tmp + r * e + j,
                                  __riscv_vfwcvt_f_f_v_f32m4(
                                      __riscv_vle16_v_f16m2(input + size_t(row) * e + j, vl), vl),
                                  vl);
            j += vl;
          }
          norm_f32(tmp + r * e, p.attn_norm, e, p.epsilon, tmp + r * e);
        }
        if (debug)
          std::memcpy((after ? debug->norm_ffn : debug->norm_attn).data() + size_t(row) * e,
                      tmp + r * e, size_t(e) * 4);
      }
      if (full)
        pack_full_i8(tmp, e, rows, e, ap.data() + size_t(first / 16) * 16 * e,
                     ascale.data() + first, after ? p.ffn_smooth : p.input_smooth);
      else if (radix) {
        if (merged)
          prepare_s8(tmp, e, e, rows, ap.data() + size_t(first / 8) * (e / 256) * ast);
        else if (!ime_i4_microkernel::prepare(tmp, e, 256,
                                              ap.data() + size_t(first / 8) * (e / 256) * ast, 2,
                                              true, true, true, 256, rows))
          std::abort();
      } else
        pack16_rows(tmp, e, rows, e,
                    reinterpret_cast<Half *>(ap.data()) + size_t(first / 16) * 16 * e);
    }
    if (!after)
      for (int r = t; r < m; r += nt)
        for (int j = 0; j < 64; ++j) {
          float theta =
              float(positions ? positions[(mrope && j < 60 ? j % 3 : 0) * m + r] : r)
              * std::pow(p.rope_base, -float(2 * j) / 128);
          cs[size_t(r) * 64 + j] = std::cos(theta);
          sn[size_t(r) * 64 + j] = std::sin(theta);
        }
  }
  void norm_rope(const float *src, const float *norm, int row, Half *dst, float *snap_norm,
                 float *snap_rope) {
    alignas(128) float x[128];
    norm_f32(src, norm, 128, p.epsilon, x);
    if (snap_norm)
      std::memcpy(snap_norm, x, 512);
    for (int j = 0; j < 64; j += 64) {
      const size_t vl = 64;
      auto lo = __riscv_vle32_v_f32m2(x + j, vl), hi = __riscv_vle32_v_f32m2(x + 64 + j, vl);
      auto co = __riscv_vle32_v_f32m2(cs.data() + size_t(row) * 64 + j, vl),
           si = __riscv_vle32_v_f32m2(sn.data() + size_t(row) * 64 + j, vl);
      auto a = __riscv_vfsub_vv_f32m2(__riscv_vfmul_vv_f32m2(lo, co, vl),
                                      __riscv_vfmul_vv_f32m2(hi, si, vl), vl);
      auto b = __riscv_vfadd_vv_f32m2(__riscv_vfmul_vv_f32m2(hi, co, vl),
                                      __riscv_vfmul_vv_f32m2(lo, si, vl), vl);
      __riscv_vse16_v_f16m1(dst + j, __riscv_vfncvt_f_f_w_f16m1(a, vl), vl);
      __riscv_vse16_v_f16m1(dst + 64 + j, __riscv_vfncvt_f_f_w_f16m1(b, vl), vl);
    }
    if (snap_rope)
      for (int j = 0; j < 128; ++j)
        snap_rope[j] = float(dst[j]);
  }
  void attention_head(int t, int kh) {
    const size_t kvbytes = size_t(sp) * 128 * 4;
    auto *pk = reinterpret_cast<Half *>(tcm[t]);
    auto *pv = pk + size_t(sp) * 128;
    std::fill_n(pk, size_t(sp) * 128 * 2, Half(0));
    // Materialize only one current-sequence KV head, in TCM; no KV cache.
    const int project_rows =
        std::max(16, std::min(128, int((TCM - kvbytes - 100000) / (128 * 4)) / 16 * 16));
    for (int first = 0; first < m; first += project_rows) {
      const int rows = std::min(project_rows, m - first), rp = round_up(rows, 16);
      auto *tmp = reinterpret_cast<float *>(tcm[t] + kvbytes);
      const size_t occupied = kvbytes + size_t(rp) * 128 * 4;
      project(p.k, first, rows, kh * 128, 128, tmp, 128, t, occupied);
      alignas(128) Half line[128];
      for (int r = 0; r < rows; ++r) {
        int row = first + r;
        norm_rope(tmp + r * 128, p.k_norm, row, line,
                  debug ? debug->k_norm.data() + (size_t(row) * hk + kh) * 128 : nullptr,
                  debug ? debug->k_rope.data() + (size_t(kh) * m + row) * 128 : nullptr);
        for (int j = 0; j < 128; j += 8)
          std::memcpy(pk + size_t(row / 32) * 128 * 32 + (j / 8) * 256 + (row % 32) * 8, line + j,
                      16);
      }
      project(p.v, first, rows, kh * 128, 128, tmp, 128, t, occupied);
      for (int r = 0; r < rows; ++r)
        for (int j = 0; j < 128; j += 32) {
          int row = first + r;
          auto value = __riscv_vfncvt_f_f_w_f16m1(__riscv_vle32_v_f32m2(tmp + r * 128 + j, 32), 32);
          __riscv_vsse16_v_f16m1(pv + size_t(j / 32) * sp * 32 + (row / 8) * 256 + (j % 32) * 8 +
                                     row % 8,
                                 16, value, 32);
          if (debug)
            __riscv_vse32_v_f32m2(debug->v_heads.data() + (size_t(kh) * m + row) * 128 + j,
                                  __riscv_vfwcvt_f_f_v_f32m2(value, 32), 32);
        }
    }
    int maxrows = std::min(128, int((TCM - kvbytes - 100000) / (256 * 4)) / 16 * 16);
    maxrows = std::max(16, maxrows);
    for (int qp = kh * (h / hk) / 2; qp < (kh + 1) * (h / hk) / 2; ++qp)
      for (int first = 0; first < m; first += maxrows) {
        const int rows = std::min(maxrows, m - first), rp = round_up(rows, 16);
        auto *qf = reinterpret_cast<float *>(tcm[t] + kvbytes);
        project(p.q, first, rows, qp * 256, 256, qf, 256, t, kvbytes + size_t(rp) * 256 * 4);
        auto *qh = reinterpret_cast<Half *>(qf);
        for (int r = 0; r < rows; ++r)
          for (int hh = 0; hh < 2; ++hh)
            norm_rope(qf + size_t(r) * 256 + hh * 128, p.q_norm, first + r,
                      qh + size_t(r) * 256 + hh * 128,
                      debug ? debug->q_norm.data() + (size_t(first + r) * h + qp * 2 + hh) * 128
                            : nullptr,
                      debug ? debug->q_rope.data() + (size_t(qp * 2 + hh) * m + first + r) * 128
                            : nullptr);
        Arena ar{tcm[t], kvbytes + size_t(rp) * 256 * 2, TCM};
        auto *qa = (Half *)ar.get(16 * 128 * 2);
        auto *prob = (Half *)ar.get(size_t(16) * sp * 2);
        auto *score = (float *)ar.get(size_t(16) * sp * 4);
        auto *tile = (float *)ar.get(2048);
        auto *ctx = (float *)ar.get(16 * 256 * 4);
        for (int r0 = 0; r0 < rows; r0 += 16) {
          const int mr = std::min(16, rows - r0);
          std::fill_n(ctx, 16 * 256, 0.f);
          for (int hh = 0; hh < 2; ++hh) {
            std::fill_n(qa, 16 * 128, Half(0));
            std::fill_n(prob, size_t(16) * sp, Half(0));
            for (int r = 0; r < mr; ++r)
              for (int j = 0; j < 128; j += 8)
                std::memcpy(qa + (j / 8) * 128 + r * 8, qh + size_t(r0 + r) * 256 + hh * 128 + j,
                            16);
            for (int key = 0; key < sp; key += 32) {
              radix_attention::ime_tile(qa, pk + size_t(key) * 128, 16, tile);
              for (int r = 0; r < mr; ++r)
                __riscv_vse32_v_f32m1(
                    score + r * sp + key,
                    __riscv_vfmul_vf_f32m1(__riscv_vle32_v_f32m1(tile + r * 32, 32),
                                           0.08838834764831845f, 32),
                    32);
            }
            for (int r = 0; r < mr; ++r) {
              const int query = first + r0 + r;
              float mx = -INFINITY;
              for (int j = 0; j < m; ++j) {
                if (j > query)
                  score[r * sp + j] = -INFINITY;
                else if (mask)
                  score[r * sp + j] += mask[size_t(query) * m + j];
                mx = std::max(mx, score[r * sp + j]);
              }
              if (mx == -INFINITY)
                continue;
              float sum = 0;
              for (int j = 0; j < m;) {
                size_t vl = __riscv_vsetvl_e32m2(m - j);
                auto values = __riscv_vle32_v_f32m2(score + r * sp + j, vl);
                auto ex = radix_attention::expv(__riscv_vfsub_vf_f32m2(values, mx, vl), vl);
                ex = __riscv_vfmerge_vfm_f32m2(
                    ex, 0, __riscv_vmfeq_vf_f32m2_b16(values, -INFINITY, vl), vl);
                __riscv_vse32_v_f32m2(score + r * sp + j, ex, vl);
                sum += __riscv_vfmv_f_s_f32m1_f32(
                    __riscv_vfredusum_vs_f32m2_f32m1(ex, __riscv_vfmv_v_f_f32m1(0, 1), vl));
                j += vl;
              }
              for (int j = 0; j < m; j += 8) {
                size_t vl = std::min(8, m - j);
                auto v =
                    __riscv_vfdiv_vf_f32m2(__riscv_vle32_v_f32m2(score + r * sp + j, vl), sum, vl);
                auto ph = __riscv_vfncvt_f_f_w_f16m1(v, vl);
                if (debug)
                  __riscv_vse32_v_f32m2(debug->probabilities.data() +
                                            (size_t(qp * 2 + hh) * m + query) * m + j,
                                        __riscv_vfwcvt_f_f_v_f32m2(ph, vl), vl);
                // Preserve already-rounded FP16 subnormals on the IME input.
                ph = __riscv_vfmul_vf_f16m1(ph, Half(1024), vl);
                __riscv_vse16_v_f16m1(prob + (j / 8) * 128 + r * 8, ph, vl);
              }
            }
            for (int col = 0; col < 128; col += 32) {
              radix_attention::ime_tile(prob, pv + size_t(col) * sp, sp / 8, tile);
              for (int r = 0; r < mr; ++r)
                __riscv_vse32_v_f32m1(ctx + r * 256 + hh * 128 + col,
                                      __riscv_vfmul_vf_f32m1(
                                          __riscv_vle32_v_f32m1(tile + r * 32, 32), 1.f / 1024, 32),
                                      32);
            }
          }
          if (debug)
            for (int r = 0; r < mr; ++r)
              std::memcpy(debug->attention_context.data() + size_t(first + r0 + r) * h * 128 +
                              qp * 256,
                          ctx + r * 256, 1024);
          // Directly produce O's FP16 IME layout, without an NHWC DDR context.
          if (full)
            for (int r = 0; r < mr; ++r)
              std::memcpy(residual.data() + size_t(first + r0 + r) * e + qp * 256, ctx + r * 256,
                          1024);
          else {
            auto *dst =
                op.data() + size_t((first + r0) / 16) * 16 * h * 128 + size_t(qp) * 256 * 16;
            pack16_rows(ctx, 256, mr, 256, dst);
          }
        }
      }
  }
  void attention(int t) {
    for (int kh = t; kh < hk; kh += nt)
      attention_head(t, kh);
  }
  void project_o(int t) {
    const int cb = (e / 32) * t / nt * 32, ce = (e / 32) * (t + 1) / nt * 32;
    if (cb == ce)
      return;
    for (int first = 0; first < m; first += 128) {
      const int rows = std::min(128, m - first);
      if (full)
        linear_full_i8(p.o, ap.data(), ascale.data(), rows, first, cb, ce - cb,
                       residual.data() + size_t(first) * e + cb, e, tcm[t], TCM);
      else
        linear16(p.o, op.data(), rows, first, cb, ce - cb, residual.data() + size_t(first) * e + cb,
                 e, tcm[t], TCM);
      for (int r = 0; r < rows; ++r)
        for (int c = cb; c < ce;) {
          size_t vl = __riscv_vsetvl_e32m4(ce - c);
          auto *dst = residual.data() + size_t(first + r) * e + c;
          auto x = __riscv_vle32_v_f32m4(dst, vl);
          if (debug)
            __riscv_vse32_v_f32m4(debug->attention_out.data() + size_t(first + r) * e + c, x, vl);
          auto in = __riscv_vfwcvt_f_f_v_f32m4(
              __riscv_vle16_v_f16m2(input + size_t(first + r) * e + c, vl), vl);
          x = __riscv_vfadd_vv_f32m4(x, in, vl);
          __riscv_vse32_v_f32m4(dst, x, vl);
          if (debug)
            __riscv_vse32_v_f32m4(debug->residual_attn.data() + size_t(first + r) * e + c, x, vl);
          c += vl;
        }
    }
  }
  static vfloat32m2_t reciprocal(vfloat32m2_t x, size_t vl) {
    auto y = __riscv_vfrec7_v_f32m2(x, vl);
    for (int j = 0; j < 2; ++j)
      y = __riscv_vfmul_vv_f32m2(y,
                                 __riscv_vfsub_vv_f32m2(__riscv_vfmv_v_f_f32m2(2, vl),
                                                        __riscv_vfmul_vv_f32m2(x, y, vl), vl),
                                 vl);
    return y;
  }
  int resident_rows() const {
    if (!radix || (p.gate.type != WeightType::PackedG256 && p.gate.type != WeightType::PackedS8) ||
        (nt != 1 && nt % 2))
      return 0;
    for (int rows = 128; rows >= 32; rows -= 32)
      if (size_t(rows / 16) * (e / 256) * ast + size_t(rows) * 160 * 4 + size_t(e / 256) * 8704 +
              8192 <=
          TCM)
        return rows;
    return 0;
  }
  void resident_gu(int t, int batch) {
    const size_t astride = size_t(e / 256) * ast, bst = size_t(e / 256) * 8704;
    const int pair = nt == 1 ? 0 : t / 2, side = nt == 1 ? 0 : t % 2;
    for (int first = 0; first < m; first += batch) {
      const int rows = std::min(batch, m - first), rp = round_up(rows, 32), nr = rp / 8,
                half = nr / 2;
      const size_t abytes = size_t(half) * astride;
      auto *local = tcm[t];
      auto *g = reinterpret_cast<float *>(local + abytes);
      auto *u = g + size_t(nr) * 4 * 256;
      auto *b = reinterpret_cast<uint8_t *>(u + size_t(nr) * 256);
      auto aptr = [&](int r) {
        return tcm[nt == 1 ? (r / half) : (pair * 2 + r / half)] + size_t(r % half) * astride;
      };
      for (int r = side * half; r < (nt == 1 ? nr : (side + 1) * half); ++r) {
        if (r * 8 < rows)
          fp4_residual::copy_gu_input(aptr(r), ap.data() + size_t((first / 8) + r) * astride,
                                      astride);
        else
          std::memset(aptr(r), 0, astride);
      }
      sync.wait();
      for (int col = t * 128; col < f; col += nt * 128) {
        for (int c = 0; c < 4; ++c) {
          auto *go = g + size_t(c) * nr * 256;
          for (int which = 0; which < 2; ++which) {
            auto *dest = which ? u : go;
            const auto &w = which ? p.up : p.gate;
            weight_i4(w, col + c * 32, b);
            if (merged)
              for (int r = 0; r < nr; r += 2)
                merged_m16n32(aptr(r), aptr(r + 1), b, e / 256, dest + r * 256,
                              dest + (r + 1) * 256);
            else
              for (int r = 0; r < nr; r += 2)
                radix_m16n32(aptr(r), aptr(r + 1), b, e / 256, dest + r * 256, dest + (r + 1) * 256,
                             false);
          }
          if (debug)
            for (int r = 0; r < nr; ++r) {
              alignas(128) float tile[256];
              for (int which = 0; which < 2; ++which) {
                ime_i4_microkernel::unpack_tile((which ? u : go) + r * 256, tile, 32);
                auto &v = which ? debug->up : debug->gate;
                for (int rr = 0; rr < std::min(8, rows - r * 8); ++rr)
                  std::memcpy(v.data() + size_t(first + r * 8 + rr) * f + col + c * 32,
                              tile + rr * 32, 128);
              }
            }
          for (int i = 0; i < nr * 256;) {
            size_t vl = __riscv_vsetvl_e32m2(nr * 256 - i);
            auto a = __riscv_vle32_v_f32m2(go + i, vl), v = __riscv_vle32_v_f32m2(u + i, vl);
            auto ex = radix_attention::expv(__riscv_vfneg_v_f32m2(a, vl), vl);
            auto z = __riscv_vfmul_vv_f32m2(
                __riscv_vfmul_vv_f32m2(a, reciprocal(__riscv_vfadd_vf_f32m2(ex, 1, vl), vl), vl), v,
                vl);
            __riscv_vse32_v_f32m2(go + i, z, vl);
            i += vl;
          }
        }
        auto *line = reinterpret_cast<float *>(b);
        for (int r = 0; r < nr; r += 2) {
          for (int c = 0; c < 4; ++c)
            for (int rr = 0; rr < 2; ++rr)
              ime_i4_microkernel::unpack_tile(g + (size_t(c) * nr + r + rr) * 256,
                                              line + rr * 8 * 128 + c * 32, 128);
          const int valid = std::min(16, rows - r * 8);
          if (valid <= 0)
            continue;
          if (debug)
            for (int rr = 0; rr < valid; ++rr)
              std::memcpy(debug->swiglu.data() + size_t(first + r * 8 + rr) * f + col,
                          line + rr * 128, 512);
          pack16_rows(line, 128, valid, 128,
                      zp.data() + size_t((first + r * 8) / 16) * 16 * f + size_t(col) * 16);
        }
      }
      sync.wait();
    }
  }
  void prepare_o(int t) {
    for (int first = t * 16; first < m; first += nt * 16)
      pack_full_i8(residual.data() + size_t(first) * e, e, std::min(16, m - first), e,
                   ap.data() + size_t(first / 16) * 16 * e, ascale.data() + first, p.o_smooth);
  }
  void prepare_down(int t) {
    auto *src = swiglu_full.data();
    for (int first = t * 16; first < m; first += nt * 16)
      pack_full_i8(src + size_t(first) * f, f, std::min(16, m - first), f,
                   ap.data() + size_t(first / 16) * 16 * f, ascale.data() + first, p.down_smooth);
  }
  void fused_full_gu(int t) {
    for (int job = t; job < ((m + 127) / 128) * (f / 64); job += nt) {
      int first = job / (f / 64) * 128, col = job % (f / 64) * 64, rows = std::min(128, m - first),
          rp = round_up(rows, 16);
      auto *g = reinterpret_cast<float *>(tcm[t]);
      auto *u = g + size_t(rp) * 64;
      size_t used = size_t(rp) * 64 * 8;
      project(p.gate, first, rows, col, 64, g, 64, t, used);
      project(p.up, first, rows, col, 64, u, 64, t, used);
      for (int r = 0; r < rows; ++r) {
        auto a = __riscv_vle32_v_f32m2(g + r * 64, 64), v = __riscv_vle32_v_f32m2(u + r * 64, 64);
        if (debug) {
          __riscv_vse32_v_f32m2(debug->gate.data() + size_t(first + r) * f + col, a, 64);
          __riscv_vse32_v_f32m2(debug->up.data() + size_t(first + r) * f + col, v, 64);
        }
        auto ex = radix_attention::expv(__riscv_vfneg_v_f32m2(a, 64), 64);
        auto z = __riscv_vfmul_vv_f32m2(
            __riscv_vfmul_vv_f32m2(a, reciprocal(__riscv_vfadd_vf_f32m2(ex, 1, 64), 64), 64), v,
            64);
        __riscv_vse32_v_f32m2(swiglu_full.data() + size_t(first + r) * f + col, z, 64);
        if (debug)
          __riscv_vse32_v_f32m2(debug->swiglu.data() + size_t(first + r) * f + col, z, 64);
      }
    }
  }
  void fused_gu(int t) {
    if (full) {
      fused_full_gu(t);
      return;
    }
    if (int batch = resident_rows()) {
      resident_gu(t, batch);
      return;
    }
    for (int job = t; job < ((m + 127) / 128) * (f / 256); job += nt) {
      const int first = (job / (f / 256)) * 128, col = (job % (f / 256)) * 256;
      const int rows = std::min(128, m - first), rp = round_up(rows, 16);
      auto *g = reinterpret_cast<float *>(tcm[t]);
      auto *u = g + size_t(rp) * 256;
      const size_t used = size_t(rp) * 256 * 8;
      project(p.gate, first, rows, col, 256, g, 256, t, used);
      project(p.up, first, rows, col, 256, u, 256, t, used);
      if (debug)
        for (int r = 0; r < rows; ++r) {
          std::memcpy(debug->gate.data() + size_t(first + r) * f + col, g + r * 256, 1024);
          std::memcpy(debug->up.data() + size_t(first + r) * f + col, u + r * 256, 1024);
        }
      for (int i = 0; i < rows * 256;) {
        size_t vl = __riscv_vsetvl_e32m2(rows * 256 - i);
        auto gate = __riscv_vle32_v_f32m2(g + i, vl), up = __riscv_vle32_v_f32m2(u + i, vl);
        auto ex = radix_attention::expv(__riscv_vfneg_v_f32m2(gate, vl), vl);
        auto z = __riscv_vfmul_vv_f32m2(
            __riscv_vfmul_vv_f32m2(gate, reciprocal(__riscv_vfadd_vf_f32m2(ex, 1, vl), vl), vl), up,
            vl);
        __riscv_vse32_v_f32m2(g + i, z, vl);
        i += vl;
      }
      if (debug)
        for (int r = 0; r < rows; ++r)
          std::memcpy(debug->swiglu.data() + size_t(first + r) * f + col, g + r * 256, 1024);
      for (int r = 0; r < rows; r += 16)
        pack16_rows(g + r * 256, 256, std::min(16, rows - r), 256,
                    zp.data() + size_t((first + r) / 16) * 16 * f + size_t(col) * 16);
    }
  }
  void down(int t) {
    const int cb = (e / 32) * t / nt * 32, ce = (e / 32) * (t + 1) / nt * 32;
    if (cb == ce)
      return;
    for (int first = 0; first < m; first += 128) {
      const int rows = std::min(128, m - first);
      // One DDR workspace reused for residual+final output; Down tile results
      // are accumulated in a bounded per-thread temporary, then epilogued.
      const int width = std::min(128, ce - cb);
      auto *tmp = reinterpret_cast<float *>(tcm[t]);
      const size_t used = size_t(round_up(rows, 16)) * width * 4;
      for (int c = cb; c < ce; c += width) {
        const int n = std::min(width, ce - c);
        if (full) {
          linear_full_i8(p.down, ap.data(), ascale.data(), rows, first, c, n, tmp, width,
                         tcm[t] + used, TCM - used);
          for (int r = 0; r < rows; ++r)
            for (int j = 0; j < n;) {
              size_t vl = __riscv_vsetvl_e32m4(n - j);
              size_t ix = size_t(first + r) * e + c + j;
              auto z = __riscv_vle32_v_f32m4(tmp + r * width + j, vl);
              if (debug)
                __riscv_vse32_v_f32m4(debug->down.data() + ix, z, vl);
              z = __riscv_vfadd_vv_f32m4(z, __riscv_vle32_v_f32m4(residual.data() + ix, vl), vl);
              auto y = __riscv_vfncvt_f_f_w_f16m2(z, vl);
              __riscv_vse16_v_f16m2(output + ix, y, vl);
              if (debug)
                __riscv_vse32_v_f32m4(debug->output.data() + ix, __riscv_vfwcvt_f_f_v_f32m4(y, vl),
                                      vl);
              j += vl;
            }
          continue;
        }
        linear16_raw(p.down, zp.data(), rows, first, c, n, tmp, tcm[t] + used, TCM - used);
        alignas(128) float line[256];
        const float inv = 1.f / p.down.factor;
        for (int nn = 0; nn < n; nn += 32)
          for (int r = 0; r < rows; r += 8) {
            ime_i4_microkernel::unpack_tile(
                tmp + size_t(nn / 32) * round_up(rows, 16) * 32 + size_t(r) * 32, line, 32);
            for (int rr = 0; rr < std::min(8, rows - r); ++rr) {
              size_t vl = std::min(32, n - nn);
              auto z = __riscv_vfmul_vf_f32m1(__riscv_vle32_v_f32m1(line + rr * 32, vl), inv, vl);
              size_t idx = size_t(first + r + rr) * e + c + nn;
              if (debug)
                __riscv_vse32_v_f32m1(debug->down.data() + idx, z, vl);
              z = __riscv_vfadd_vv_f32m1(z, __riscv_vle32_v_f32m1(residual.data() + idx, vl), vl);
              auto y = __riscv_vfncvt_f_f_w_f16mf2(z, vl);
              __riscv_vse16_v_f16mf2(output + idx, y, vl);
              if (debug)
                __riscv_vse32_v_f32m1(debug->output.data() + idx, __riscv_vfwcvt_f_f_v_f32m1(y, vl),
                                      vl);
            }
          }
      }
    }
  }
  void compute(int t) {
    double last = fp4fp8::ns() * 1e-6;
    auto mark = [&](int n) {
      sync.wait();
      double now = fp4fp8::ns() * 1e-6;
      stages[t * 6 + n] = now - last;
      last = now;
    };
    prepare(t, false);
    mark(0);
    attention(t);
    if (full) {
      sync.wait();
      prepare_o(t);
    }
    mark(1);
    project_o(t);
    mark(2);
    prepare(t, true);
    mark(3);
    fused_gu(t);
    if (full) {
      sync.wait();
      prepare_down(t);
    }
    mark(4);
    down(t);
    mark(5);
  }

public:
  std::vector<double> stages;
  // radix selects the experimentally accepted numeric policy, not a shape-
  // specific microkernel. Normal operation automatically dispatches by format.
  PrefillBlock(const BlockParameters &params, int tokens, int threads, bool use_radix = true, bool borrow_tcm = false)
      : p(params), m(tokens), mp(round_up(tokens, 16)), sp(round_up(tokens, 32)), nt(threads),
        e(params.q.k), f(params.gate.n), h(params.q.n / 128), hk(params.k.n / 128),
        radix(use_radix), merged(params.q.type == WeightType::PackedS8),
        full(params.q.type == WeightType::PackedFullI8), ast(merged ? 2560 : 3328),
        tcm(threads == 1 && use_radix && params.q.type != WeightType::PackedFullI8 ? 2 : threads,
            nullptr),
        stages(threads * 6) {
    capacity = tokens; borrowed_tcm = borrow_tcm;
    for (auto *w : {&p.q, &p.k, &p.v, &p.o, &p.gate, &p.up, &p.down})
      w->validate();
    if (m < 1 || m > 512 || nt < 1 || nt > 8 || p.head_dim != 128 || e % 256 || f % 256 ||
        e > 2048 || !hk || h % hk || (h / hk) % 2 || !p.attn_norm || !p.q_norm || !p.k_norm ||
        !p.ffn_norm)
      throw std::invalid_argument("prefill shape/config");
    if (p.q.n % 256 || p.k.n % 128 || p.v.n != p.k.n || p.k.k != e || p.v.k != e ||
        p.o.k != p.q.n || p.o.n != e || p.gate.k != e || p.up.k != e || p.up.n != f ||
        p.down.k != f || p.down.n != e || !std::isfinite(p.epsilon) || p.epsilon <= 0 ||
        !std::isfinite(p.rope_base) || p.rope_base <= 0)
      throw std::invalid_argument("inconsistent block dimensions/norm/RoPE");
    for (auto *w : {&p.q, &p.k, &p.v, &p.gate, &p.up})
      if (w->type != WeightType::PackedFullI8 && w->type != WeightType::NVFP4 &&
          w->type != (radix ? (merged ? WeightType::PackedS8 : WeightType::PackedG256)
                            : WeightType::PackedFP16))
        throw std::invalid_argument("projection precision contract");
    for (auto *w : {&p.o, &p.down})
      if (w->type != WeightType::PackedFullI8 && w->type != WeightType::BF16 &&
          w->type != WeightType::PackedFP16)
        throw std::invalid_argument("BF16 projection contract");
    if (full && (f > 8192 || p.q.n != e || !p.input_smooth || !p.o_smooth || !p.ffn_smooth ||
                 !p.down_smooth))
      throw std::invalid_argument("full INT8 configuration");
    ascale.resize(mp);
    if (full)
      for (auto *w : {&p.q, &p.k, &p.v, &p.o, &p.gate, &p.up, &p.down})
        if (w->type != WeightType::PackedFullI8 || !w->scales)
          throw std::invalid_argument("full INT8 weights/scales must match");
    if (merged)
      for (auto *w : {&p.q, &p.k, &p.v, &p.gate, &p.up})
        if (w->type != WeightType::PackedS8)
          throw std::invalid_argument("merged weights must match");
    ap.resize(full    ? size_t(mp) * std::max(e, f)
              : radix ? size_t((m + 7) / 8) * (e / 256) * ast
                      : size_t(mp) * e * 2);
    op.resize(full ? 0 : size_t(mp) * h * 128);
    zp.resize(full ? 0 : size_t(mp) * f);
    swiglu_full.resize(full ? size_t(mp) * f : 0);
    residual.resize(size_t(m) * e);
    cs.resize(size_t(m) * 64);
    sn.resize(cs.size());
    sync.nt = nt;
    spine_tcm_mem_info_t info{};
    if (__riscv_vsetvl_e8m1(1024) != 128 || spine_tcm_open_handle(nullptr) ||
        spine_tcm_mem_info(&info) || info.is_fake_tcm || info.blk_num < 8 || info.blk_size < TCM)
      throw std::runtime_error("real K3 TCM/VLEN1024 required");
    try {
      for (int t = 0; t < int(tcm.size()); ++t) {
        void *mapped = spine_tcm_mem_get(t);
        if (!mapped) throw std::runtime_error("TCM mapping");
        tcm[t] = (uint8_t *)(borrowed_tcm ? mapped : spine_tcm_mem_try_wait(t, 1000000));
        if (!tcm[t]) {
          spine_tcm_mem_free(t);
          throw std::runtime_error("TCM occupied");
        }
        spine_tcm_block_info_t bi{};
        if (spine_tcm_block_info(t, &bi) || !(bi.cpu_affinity_mask & (uint64_t(1) << (8 + t))))
          throw std::runtime_error("TCM affinity");
      }
    } catch (...) {
      release();
      throw;
    }
    pthread_barrier_init(&command, nullptr, nt + 1);
    for (int t = 0; t < nt; ++t)
      workers.emplace_back([&, t] {
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(8 + t, &set);
        if (pthread_setaffinity_np(pthread_self(), sizeof(set), &set))
          std::abort();
        std::fesetround(FE_TONEAREST);
        ++ready;
        for (;;) {
          pthread_barrier_wait(&command);
          if (stop)
            break;
          compute(t);
          pthread_barrier_wait(&command);
        }
      });
    while (ready.load() != nt)
      std::this_thread::yield();
  }
  ~PrefillBlock() {
    stop = true;
    pthread_barrier_wait(&command);
    for (auto &w : workers)
      w.join();
    pthread_barrier_destroy(&command);
    release();
  }
  size_t workspace_bytes() const {
    return ap.size() + 2 * (op.size() + zp.size()) +
           4 * (residual.size() + cs.size() + sn.size() + ascale.size() + swiglu_full.size());
  }
  void set_tokens(int tokens) {
    if (tokens < 1 || tokens > capacity) throw std::invalid_argument("token capacity");
    m=tokens; mp=round_up(tokens,16); sp=round_up(tokens,32);
  }
  void enable_mrope(bool enabled) { mrope=enabled; }
  void rebind(const BlockParameters &params) {
    const Matrix *old[] = {&p.q, &p.k, &p.v, &p.o, &p.gate, &p.up, &p.down};
    const Matrix *next[] = {&params.q, &params.k, &params.v, &params.o,
                            &params.gate, &params.up, &params.down};
    for (int i = 0; i < 7; ++i) {
      next[i]->validate();
      if (next[i]->n != old[i]->n || next[i]->k != old[i]->k ||
          next[i]->type != old[i]->type || (full && !next[i]->scales))
        throw std::invalid_argument("rebind matrix contract");
    }
    if (params.head_dim != p.head_dim || !params.attn_norm || !params.q_norm ||
        !params.k_norm || !params.ffn_norm || !std::isfinite(params.epsilon) ||
        params.epsilon <= 0 || !std::isfinite(params.rope_base) || params.rope_base <= 0 ||
        (full && (!params.input_smooth || !params.o_smooth || !params.ffn_smooth ||
                  !params.down_smooth)))
      throw std::invalid_argument("rebind block contract");
    p = params;
  }
  void enable_debug(DebugStages *d) {
    debug = d;
    if (!d)
      return;
    for (auto *v :
         {&d->norm_attn, &d->attention_out, &d->residual_attn, &d->norm_ffn, &d->down, &d->output})
      v->resize(size_t(m) * e);
    for (auto *v : {&d->q_norm, &d->q_rope, &d->attention_context})
      v->resize(size_t(m) * h * 128);
    for (auto *v : {&d->k_norm, &d->k_rope, &d->v_heads})
      v->resize(size_t(m) * hk * 128);
    for (auto *v : {&d->gate, &d->up, &d->swiglu})
      v->resize(size_t(m) * f);
    d->probabilities.resize(size_t(h) * m * m);
  }
  void run(const Half *x, Half *y, const int64_t *pos = nullptr, const float *additive = nullptr) {
    if (!x || !y ||
        (reinterpret_cast<uintptr_t>(x) < reinterpret_cast<uintptr_t>(y) + size_t(m) * e * 2 &&
         reinterpret_cast<uintptr_t>(y) < reinterpret_cast<uintptr_t>(x) + size_t(m) * e * 2))
      throw std::invalid_argument("non-overlapping FP16 IO required");
    input = x;
    output = y;
    positions = pos;
    mask = additive;
    pthread_barrier_wait(&command);
    pthread_barrier_wait(&command);
  }
};
} // namespace nvfp4_block
#ifdef NVFP4_BLOCK_BENCH_MAIN
#include <malloc.h>
#include <numeric>
using namespace nvfp4_block;
template <class T, class Allocator = std::allocator<T>>
static std::vector<T, Allocator> read_file(const std::string &name) {
  std::ifstream file(name, std::ios::binary | std::ios::ate);
  if (!file)
    throw std::runtime_error(name);
  size_t bytes = file.tellg();
  if (bytes % sizeof(T))
    throw std::runtime_error("file length");
  std::vector<T, Allocator> v(bytes / sizeof(T));
  file.seekg(0);
  file.read(reinterpret_cast<char *>(v.data()), bytes);
  if (!file)
    throw std::runtime_error("file read");
  return v;
}
template <class T> static void save_file(const std::string &name, const std::vector<T> &v) {
  std::ofstream file(name, std::ios::binary);
  file.write(reinterpret_cast<const char *>(v.data()), v.size() * sizeof(T));
  if (!file)
    throw std::runtime_error("file write " + name);
}
int main(int argc, char **argv) {
  try {
    if (argc != 7)
      throw std::runtime_error(
          "usage: bench ROOT M THREADS "
          "POLICY(0=original_FP16,1=G256_dual,2=merged_G256,3=Smooth_INT8) TRIALS PREFIX");
    setvbuf(stdout, nullptr, _IONBF, 0);
    std::fesetround(FE_TONEAREST);
    std::string root = argv[1], prefix = argv[6];
    int m = std::stoi(argv[2]), nt = std::stoi(argv[3]), mode = std::stoi(argv[4]),
        trials = std::stoi(argv[5]);
    BlockParameters p;
    Matrix *matrix[] = {&p.q, &p.k, &p.v, &p.o, &p.gate, &p.up, &p.down};
    const char *names[] = {"attn_q",   "attn_k", "attn_v",  "attn_output",
                           "ffn_gate", "ffn_up", "ffn_down"};
    int ns[] = {2048, 1024, 1024, 2048, 6144, 6144, 2048};
    int ks[] = {2048, 2048, 2048, 2048, 2048, 2048, 6144};
    std::vector<float> scales[7], smooth[7];
    AlignedVector<uint8_t> weights[7];
    size_t wbytes = 0;
    for (int j = 0; j < 7; ++j) {
      bool bf = j == 3 || j == 6;
      const bool packed = std::getenv("NVFP4_USE_PACKED");
      weights[j] = read_file<uint8_t, AlignedAllocator<uint8_t>>(
          root + (packed ? "/weights/packed/" : "/weights/raw/") + names[j] +
          (mode == 3 ? ".i8full"
           : packed  ? (bf || mode == 0 ? ".ime16"
                        : mode == 2     ? ".s8"
                                        : ".g256")
                     : (bf ? ".bf16" : ".nvfp4")));
      *matrix[j] = Matrix{ns[j],
                          ks[j],
                          mode == 3 ? WeightType::PackedFullI8
                          : packed  ? (bf || mode == 0 ? WeightType::PackedFP16
                                       : mode == 2     ? WeightType::PackedS8
                                                       : WeightType::PackedG256)
                                    : (bf ? WeightType::BF16 : WeightType::NVFP4),
                          weights[j].data(),
                          weights[j].size(),
                          256.f};
      wbytes += weights[j].size();
      if (mode == 3) {
        scales[j] = read_file<float>(root + "/weights/packed/" + names[j] + ".scale");
        if (j == 0 || j == 3 || j == 4 || j == 6) {
          smooth[j] = read_file<float>(root + "/weights/packed/" + names[j] + ".smooth");
          wbytes += smooth[j].size() * 4;
        }
        matrix[j]->scales = scales[j].data();
        wbytes += scales[j].size() * 4;
      }
    }
    if (mode == 3) {
      p.input_smooth = smooth[0].data();
      p.o_smooth = smooth[3].data();
      p.ffn_smooth = smooth[4].data();
      p.down_smooth = smooth[6].data();
    }
    std::vector<float> norm[4];
    const char *norms[] = {"attn_norm", "attn_q_norm", "attn_k_norm", "ffn_norm"};
    for (int j = 0; j < 4; ++j) {
      norm[j] = read_file<float>(root + "/weights/" + norms[j] + ".f32");
      wbytes += norm[j].size() * 4;
    }
    p.attn_norm = norm[0].data();
    p.q_norm = norm[1].data();
    p.k_norm = norm[2].data();
    p.ffn_norm = norm[3].data();
    auto source = read_file<float>(root + "/cases/prefill128/input.f32");
    std::vector<Half> x(size_t(m) * 2048), y(x.size());
    for (size_t i = 0; i < x.size(); ++i)
      x[i] = Half(source[i % source.size()]);
    const auto saved = x;
    PrefillBlock block(p, m, nt, mode != 0);
    printf("CONFIG M=%d threads=%d policy=%s resident_weight_bytes=%zu workspace_bytes=%zu "
           "FP16_IO=1 TCM_each=393216 no_persistent_KV=1 weight_conversion=according_to_format\n",
           m, nt,
           mode == 3   ? "FullINT8_Smooth"
           : mode == 2 ? "G256_merged_INT8"
           : mode      ? "G256_dual"
                       : "G16_fp16",
           wbytes, block.workspace_bytes());
    if (const char *seconds = std::getenv("NVFP4_BW_SECONDS")) {
      for (int i = 0; i < 5; ++i)
        block.run(x.data(), y.data());
      double begin = fp4fp8::ns() * 1e-9, duration = std::atof(seconds);
      int count = 0;
      do {
        block.run(x.data(), y.data());
        ++count;
      } while (fp4fp8::ns() * 1e-9 - begin < duration);
      double elapsed = fp4fp8::ns() * 1e-9 - begin;
      printf("BANDWIDTH_WORK count=%d seconds=%.9f ms=%.9f\n", count, elapsed,
             elapsed * 1000 / count);
      return 0;
    }
    DebugStages debug;
    block.enable_debug(&debug);
    block.run(x.data(), y.data());
    block.enable_debug(nullptr);
#define DUMP(x) save_file(prefix + "." #x ".f32", debug.x)
    DUMP(norm_attn);
    DUMP(q_norm);
    DUMP(k_norm);
    DUMP(q_rope);
    DUMP(k_rope);
    DUMP(v_heads);
    DUMP(probabilities);
    DUMP(attention_context);
    DUMP(attention_out);
    DUMP(residual_attn);
    DUMP(norm_ffn);
    DUMP(gate);
    DUMP(up);
    DUMP(swiglu);
    DUMP(down);
    DUMP(output);
#undef DUMP
    save_file(prefix + ".output.f16", y);
    std::vector<Half> reference = y;
    debug = DebugStages{};
    malloc_trim(0);
    bool good = true;
    for (int phase = 0; phase < 3; ++phase) {
      for (size_t i = 0; i < x.size(); ++i)
        x[i] = Half(float(saved[i]) * (phase == 0 ? -1 : phase == 1 ? 0 : 1));
      std::fill(y.begin(), y.end(), Half(NAN));
      block.run(x.data(), y.data());
      bool finite = std::all_of(y.begin(), y.end(), [](Half h) { return std::isfinite(float(h)); });
      bool zero = std::all_of(y.begin(), y.end(), [](Half h) { return float(h) == 0; });
      bool restored = phase != 2 || y == reference;
      good &= finite && restored && (phase == 1 ? zero : !zero);
      printf("CHECK phase=%d finite=%d zero=%d restored=%d\n", phase, finite, zero, restored);
      if (phase == 0) {
        std::vector<float> value(y.begin(), y.end());
        save_file(prefix + ".negative.f32", value);
      }
    }
    for (int trial = 0; trial < trials; ++trial) {
      auto start = fp4fp8::ns();
      block.run(x.data(), y.data());
      double warm = (fp4fp8::ns() - start) * 1e-6;
      int reps = std::max(1, int(150 / std::max(.01, warm)));
      start = fp4fp8::ns();
      for (int rep = 0; rep < reps; ++rep)
        block.run(x.data(), y.data());
      double ms = (fp4fp8::ns() - start) * 1e-6 / reps;
      bool stable = y == reference;
      good &= stable;
      printf("TIME trial=%d ms=%.6f repeats=%d stable=%d stages=", trial, ms, reps, stable);
      for (int stage = 0; stage < 6; ++stage) {
        double value = 0;
        for (int t = 0; t < nt; ++t)
          value = std::max(value, block.stages[t * 6 + stage]);
        printf("%s%.6f", stage ? "," : "", value);
      }
      printf("\n");
    }
    std::ifstream status("/proc/self/status");
    std::string line;
    while (std::getline(status, line))
      if (line.rfind("VmRSS:", 0) == 0 || line.rfind("VmHWM:", 0) == 0)
        printf("%s\n", line.c_str());
    printf("RESULT %s\n", good ? "PASS" : "FAIL");
    return good ? 0 : 1;
  } catch (const std::exception &e) {
    fprintf(stderr, "%s\n", e.what());
    return 1;
  }
}
#endif
