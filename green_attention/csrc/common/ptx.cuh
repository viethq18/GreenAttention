// Inline-PTX helpers for the SM90 kernels: mbarrier, cp.async, wgmma descriptors,
// packed fp8 converts, named barriers, setmaxnreg.
//
// Barrier / cp.async / descriptor idioms follow FlashAttention-3 (hopper/, BSD-3-Clause,
// Copyright (c) 2022, the respective contributors, as shown by the AUTHORS file) and
// CUTLASS (BSD-3-Clause, Copyright (c) NVIDIA CORPORATION).
#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace green {

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// ---------------------------------------------------------------- mbarrier
__device__ __forceinline__ void mbar_init(uint32_t bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(bar), "r"(count));
}

__device__ __forceinline__ void mbar_arrive(uint32_t bar) {
  asm volatile(
      "{\n\t.reg .b64 st;\n\t"
      "mbarrier.arrive.shared::cta.b64 st, [%0];\n\t}\n" ::"r"(bar)
      : "memory");
}

// try_wait loop on a phase parity (blocks until the phase with that parity completed).
__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity) {
  asm volatile(
      "{\n\t.reg .pred p;\n"
      "WAIT_%=:\n\t"
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n\t"
      "@!p bra WAIT_%=;\n\t}\n" ::"r"(bar),
      "r"(parity)
      : "memory");
}

// Arrive on `bar` (without raising its pending count) once all of this thread's prior
// cp.async operations have completed.
__device__ __forceinline__ void cp_async_mbar_arrive_noinc(uint32_t bar) {
  asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n" ::"r"(bar) : "memory");
}

// ---------------------------------------------------------------- cp.async
__device__ __forceinline__ void cp_async_16(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(src) : "memory");
}

// Make generic-proxy smem writes (cp.async, st.shared) visible to the async proxy (wgmma).
__device__ __forceinline__ void fence_proxy_async() {
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

// ---------------------------------------------------------------- programmatic dependent launch
// Let the dependent (secondary) grid start launching; its griddep_wait() still blocks until this grid completes.
__device__ __forceinline__ void griddep_launch_dependents() { asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory"); }
// Block until every primary grid this one depends on has completed and flushed its memory.
__device__ __forceinline__ void griddep_wait() { asm volatile("griddepcontrol.wait;\n" ::: "memory"); }

// ---------------------------------------------------------------- named barriers / regs
__device__ __forceinline__ void named_bar_sync(int id, int nthreads) {
  asm volatile("bar.sync %0, %1;\n" ::"r"(id), "r"(nthreads) : "memory");
}

template <int N>
__device__ __forceinline__ void reg_alloc() {
  asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" ::"n"(N));
}
template <int N>
__device__ __forceinline__ void reg_dealloc() {
  asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" ::"n"(N));
}

// ---------------------------------------------------------------- wgmma
__device__ __forceinline__ void wgmma_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wgmma_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int N>
__device__ __forceinline__ void wgmma_wait() {
  asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");
}

// Keep the compiler from touching/moving accumulator registers across async wgmma.
__device__ __forceinline__ void fence_operand(float& r) { asm volatile("" : "+f"(r)::"memory"); }
__device__ __forceinline__ void fence_operand(uint32_t& r) { asm volatile("" : "+r"(r)::"memory"); }

// 128B-swizzle, K-major smem matrix descriptor (LBO unused for swizzled K-major = 1,
// SBO = 8 rows * 128 B = 1024 B, layout type 1 = SWIZZLE_128B). `addr` is a shared address.
__device__ __forceinline__ uint64_t make_desc_sw128(uint32_t addr) {
  uint64_t d = 0;
  d |= static_cast<uint64_t>((addr >> 4) & 0x3FFFu);
  d |= static_cast<uint64_t>(1) << 16;
  d |= static_cast<uint64_t>(1024 >> 4) << 32;
  d |= static_cast<uint64_t>(1) << 62;
  return d;
}

// ---------------------------------------------------------------- fp8 / math
// Pack (lo, hi) floats into two e4m3 bytes (saturating to +-448); lo ends up in byte 0.
__device__ __forceinline__ uint32_t cvt_e4m3x2(float lo, float hi) {
  uint16_t r;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;\n" : "=h"(r) : "f"(hi), "f"(lo));
  return r;
}

// Four floats -> four e4m3 bytes, element 0 in the lowest byte.
__device__ __forceinline__ uint32_t cvt_e4m3x4(float a, float b, float c, float d) {
  return cvt_e4m3x2(a, b) | (cvt_e4m3x2(c, d) << 16);
}

__device__ __forceinline__ float ex2(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t sel) {
  uint32_t d;
  asm("prmt.b32 %0, %1, %2, %3;\n" : "=r"(d) : "r"(a), "r"(b), "r"(sel));
  return d;
}

__device__ __forceinline__ float bf16_lo_to_f32(uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float bf16_hi_to_f32(uint32_t w) { return __uint_as_float(w & 0xFFFF0000u); }

}  // namespace green
