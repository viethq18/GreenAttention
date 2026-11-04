// SPDX-License-Identifier: Apache-2.0
// GreenAttention A3: Gated-DeltaNet recurrent *decode* kernel for SM90 (H100).
//
// Semantics are those of SGLang's TritonGDNKernel.packed_decode / .decode (see
// docs/gdn-contract.md).  Per (sequence, value head) and one token:
//     g    = -exp(A_log) * softplus(a + dt_bias)        beta = sigmoid(b)
//     q'   = q / sqrt(|q|^2 + 1e-6) * scale             k'   = k / sqrt(|k|^2 + 1e-6)
//     S   *= exp(g)                                     (S is [V, K], K contiguous)
//     v'   = beta * (v - S k')                          S   += v' (x) k'
//     o    = S q'
// Re-associated so that both reductions read the *original* state:
//     S k'  -> e^g (S0 k'),   S q' -> e^g (S0 q') + v' (k'.q')
// which lets one pass over the state register file produce both dot products.
//
// Design (memory-bound: bytes = 2 x state; arithmetic is ~2 FMA per state element):
//   * one warp per CTA, R state rows (of the V dimension) per CTA, grid (V/R, B*HV);
//   * every lane holds VEC consecutive K-columns of NJ rows; all NJ vector loads are
//     issued before any math, giving R*K*sizeof(state) bytes in flight per warp;
//   * dot products over K are reduce-scattered across the LPR lanes of a row group so
//     that every row ends up owned by exactly one lane (about 2*NJ shuffles in total
//     rather than 2*NJ*log2(LPR));
//   * state dtype (fp32 | bf16) is a template parameter: storage in that dtype, math fp32.
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace green {
namespace gdn {

constexpr int kH = 16;    // q/k heads
constexpr int kHV = 48;   // value heads
constexpr int kK = 128;   // head_k_dim
constexpr int kV = 128;   // head_v_dim
constexpr unsigned kFull = 0xffffffffu;

__device__ __forceinline__ float bf2f(__nv_bfloat16 x) { return __bfloat162float(x); }

__device__ __forceinline__ float ldf(const void* p, int bf16, int64_t i) {
  return bf16 ? bf2f(static_cast<const __nv_bfloat16*>(p)[i]) : static_cast<const float*>(p)[i];
}

__device__ __forceinline__ float softplus_gate(float a, float dt_bias, float A_log) {
  const float x = a + dt_bias;
  const float sp = x <= 20.0f ? logf(1.0f + expf(x)) : x;
  return -expf(A_log) * sp;
}

// ---------------------------------------------------------------------------------------
// State vector load/store.  ST = float: 4 columns / 16 B, ST = bf16: 8 columns / 16 B.
// (ld.global.cs / st.global.cs streaming hints measured 0-3% slower at bs 32/51: not used.)
// ---------------------------------------------------------------------------------------
__device__ __forceinline__ float4 ld16(const float* p) { return *reinterpret_cast<const float4*>(p); }
__device__ __forceinline__ void st16(float* p, float4 v) { *reinterpret_cast<float4*>(p) = v; }
__device__ __forceinline__ uint4 ld16u(const __nv_bfloat16* p) { return *reinterpret_cast<const uint4*>(p); }
__device__ __forceinline__ void st16u(__nv_bfloat16* p, uint4 v) { *reinterpret_cast<uint4*>(p) = v; }
template <typename ST>
struct StateVec;

template <>
struct StateVec<float> {
  static constexpr int VEC = 4;
  __device__ __forceinline__ static void load(const float* p, float (&r)[4]) {
    const float4 t = ld16(p);
    r[0] = t.x; r[1] = t.y; r[2] = t.z; r[3] = t.w;
  }
  __device__ __forceinline__ static void store(float* p, const float (&r)[4]) {
    st16(p, make_float4(r[0], r[1], r[2], r[3]));
  }
};

template <>
struct StateVec<__nv_bfloat16> {
  static constexpr int VEC = 8;
  __device__ __forceinline__ static void load(const __nv_bfloat16* p, float (&r)[8]) {
    const uint4 t = ld16u(p);
    const uint32_t w[4] = {t.x, t.y, t.z, t.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      r[2 * i] = __uint_as_float(w[i] << 16);
      r[2 * i + 1] = __uint_as_float(w[i] & 0xffff0000u);
    }
  }
  __device__ __forceinline__ static void store(__nv_bfloat16* p, const float (&r)[8]) {
    uint32_t w[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      __nv_bfloat162 t = __floats2bfloat162_rn(r[2 * i], r[2 * i + 1]);
      w[i] = *reinterpret_cast<uint32_t*>(&t);
    }
    st16u(p, make_uint4(w[0], w[1], w[2], w[3]));
  }
};

// q/k slice of VEC bf16 values per lane (8 B or 16 B load).
template <int VEC>
__device__ __forceinline__ void load_bf16_slice(const __nv_bfloat16* p, float (&r)[VEC]) {
  if constexpr (VEC == 4) {
    const uint2 t = *reinterpret_cast<const uint2*>(p);
    const uint32_t w[2] = {t.x, t.y};
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      r[2 * i] = __uint_as_float(w[i] << 16);
      r[2 * i + 1] = __uint_as_float(w[i] & 0xffff0000u);
    }
  } else {
    const uint4 t = *reinterpret_cast<const uint4*>(p);
    const uint32_t w[4] = {t.x, t.y, t.z, t.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      r[2 * i] = __uint_as_float(w[i] << 16);
      r[2 * i + 1] = __uint_as_float(w[i] & 0xffff0000u);
    }
  }
}

// ---------------------------------------------------------------------------------------
// Butterfly reduce-scatter of NJ (x, y) pairs over the 2^LB lanes of a row group.
// Lane c ends with the full sum of pair index (c & (NJ-1)) (valid for c < NJ).
// ---------------------------------------------------------------------------------------
template <int NJ, int LB>
__device__ __forceinline__ void reduce_scatter_pairs(float (&a)[NJ], float (&b)[NJ], int c, float& ra,
                                                     float& rb) {
  constexpr int JB = __builtin_ctz(NJ);
  static_assert((1 << JB) == NJ && NJ <= (1 << LB), "NJ must be a power of two <= lanes per row");
#pragma unroll
  for (int s = JB - 1; s >= 0; --s) {
    const int half = 1 << s;
    const bool hi = (c >> s) & 1;
#pragma unroll
    for (int i = 0; i < half; ++i) {
      const float sa = hi ? a[i] : a[i + half];
      const float ka = hi ? a[i + half] : a[i];
      const float sb = hi ? b[i] : b[i + half];
      const float kb = hi ? b[i + half] : b[i];
      a[i] = ka + __shfl_xor_sync(kFull, sa, half);
      b[i] = kb + __shfl_xor_sync(kFull, sb, half);
    }
  }
  float x = a[0], y = b[0];
#pragma unroll
  for (int s = JB; s < LB; ++s) {
    x += __shfl_xor_sync(kFull, x, 1 << s);
    y += __shfl_xor_sync(kFull, y, 1 << s);
  }
  ra = x;
  rb = y;
}

// ---------------------------------------------------------------------------------------
// Decode
// ---------------------------------------------------------------------------------------
struct DecodeParams {
  const __nv_bfloat16* q;  // token 0, head 0
  const __nv_bfloat16* k;
  const __nv_bfloat16* v;
  int64_t q_ts, k_ts, v_ts;  // token strides (elements)
  const __nv_bfloat16* a;
  const __nv_bfloat16* b;
  int64_t a_ts, b_ts;
  const float* A_log;
  const void* dt_bias;  // fp32 or bf16
  int dt_bias_bf16;
  const void* cache_idx;  // int32 or int64
  int idx64;
  const int* cu_seqlens;  // null for the packed form
  void* state;
  int64_t state_slot_stride;
  __nv_bfloat16* out;  // [B, HV, V]
  float scale;
};

// PACKED = TritonGDNKernel.packed_decode semantics (beta rounded to bf16, PAD -> zero output),
// !PACKED = TritonGDNKernel.decode semantics (plain fp32 beta, PAD -> zero state, output computed).
template <typename ST, int R, bool PACKED>
__global__ void __launch_bounds__(32) decode_kernel(const DecodeParams p) {
  using SV = StateVec<ST>;
  constexpr int VEC = SV::VEC;
  constexpr int LPR = kK / VEC;
  constexpr int LB = __builtin_ctz(LPR);
  constexpr int G = 32 / LPR;
  constexpr int NJ = R / G;
  static_assert(R % G == 0, "R must be a multiple of the row groups per warp");

  const int lane = threadIdx.x;
  const int c = lane & (LPR - 1);
  const int g = lane / LPR;
  const int n = blockIdx.y / kHV;
  const int hv = blockIdx.y - n * kHV;
  const int h = hv / (kHV / kH);
  const int row0 = blockIdx.x * R;

  int64_t bos = n;
  if (p.cu_seqlens != nullptr) {
    const int b0 = p.cu_seqlens[n];
    const int b1 = p.cu_seqlens[n + 1];
    bos = b0;
    if (b1 - b0 != 1) {  // empty (graph padding) sequence: no state update, zero output
      if (lane < R) p.out[(bos * kHV + hv) * kV + row0 + lane] = __float2bfloat16_rn(0.f);
      return;
    }
  }
  const int64_t slot = p.idx64 ? static_cast<const int64_t*>(p.cache_idx)[n]
                               : static_cast<int64_t>(static_cast<const int32_t*>(p.cache_idx)[n]);
  const bool valid = slot >= 0;
  if (PACKED && !valid) {
    if (lane < R) p.out[(bos * kHV + hv) * kV + row0 + lane] = __float2bfloat16_rn(0.f);
    return;
  }

  // 1. issue every state load first (the only large transfer)
  float S[NJ][VEC];
  ST* sbase = nullptr;
  if (valid) {
    sbase = static_cast<ST*>(p.state) + slot * p.state_slot_stride + (int64_t)hv * kV * kK +
            (int64_t)(row0 + g) * kK + c * VEC;
#pragma unroll
    for (int j = 0; j < NJ; ++j) SV::load(sbase + (int64_t)j * G * kK, S[j]);
  } else {
#pragma unroll
    for (int j = 0; j < NJ; ++j)
#pragma unroll
      for (int e = 0; e < VEC; ++e) S[j][e] = 0.f;
  }

  // 2. q / k slices, norms, gating
  float qv[VEC], kv[VEC];
  load_bf16_slice<VEC>(p.q + bos * p.q_ts + h * kK + c * VEC, qv);
  load_bf16_slice<VEC>(p.k + bos * p.k_ts + h * kK + c * VEC, kv);
  float qq = 0.f, kk = 0.f, kq = 0.f;
#pragma unroll
  for (int e = 0; e < VEC; ++e) {
    qq = fmaf(qv[e], qv[e], qq);
    kk = fmaf(kv[e], kv[e], kk);
    kq = fmaf(kv[e], qv[e], kq);
  }
#pragma unroll
  for (int s = 0; s < LB; ++s) {
    qq += __shfl_xor_sync(kFull, qq, 1 << s);
    kk += __shfl_xor_sync(kFull, kk, 1 << s);
    kq += __shfl_xor_sync(kFull, kq, 1 << s);
  }
  const float qn_s = 1.0f / sqrtf(qq + 1e-6f) * p.scale;  // scale folded into q normalization
  const float kn_s = 1.0f / sqrtf(kk + 1e-6f);
  const float kq_n = kq * qn_s * kn_s;  // k'.q'
#pragma unroll
  for (int e = 0; e < VEC; ++e) {
    qv[e] *= qn_s;
    kv[e] *= kn_s;
  }

  const float a_val = bf2f(p.a[bos * p.a_ts + hv]);
  const float b_val = bf2f(p.b[bos * p.b_ts + hv]);
  const float dtb = ldf(p.dt_bias, p.dt_bias_bf16, hv);
  const float eg = expf(softplus_gate(a_val, dtb, p.A_log[hv]));
  float beta = 1.0f / (1.0f + expf(-b_val));
  if (PACKED) beta = bf2f(__float2bfloat16_rn(beta));

  // 3. both reductions on the original state
  float rk[NJ], rq[NJ];
#pragma unroll
  for (int j = 0; j < NJ; ++j) {
    float sk = 0.f, sq = 0.f;
#pragma unroll
    for (int e = 0; e < VEC; ++e) {
      sk = fmaf(S[j][e], kv[e], sk);
      sq = fmaf(S[j][e], qv[e], sq);
    }
    rk[j] = sk;
    rq[j] = sq;
  }
  float Sk, Sq;
  reduce_scatter_pairs<NJ, LB>(rk, rq, c, Sk, Sq);

  // 4. owner lane (c < NJ) owns row row0 + c*G + g
  float vnew = 0.f;
  if (c < NJ) {
    const int row = row0 + c * G + g;
    const float v = bf2f(p.v[bos * p.v_ts + hv * kV + row]);
    vnew = beta * (v - eg * Sk);
    p.out[(bos * kHV + hv) * kV + row] = __float2bfloat16_rn(fmaf(vnew, kq_n, eg * Sq));
  }

  // 5. S = e^g S + vnew (x) k'
  if (valid) {
#pragma unroll
    for (int j = 0; j < NJ; ++j) {
      const float vn = __shfl_sync(kFull, vnew, g * LPR + j);
#pragma unroll
      for (int e = 0; e < VEC; ++e) S[j][e] = fmaf(vn, kv[e], eg * S[j][e]);
      SV::store(sbase + (int64_t)j * G * kK, S[j]);
    }
  }
}

}  // namespace gdn
}  // namespace green
