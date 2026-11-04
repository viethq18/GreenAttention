// SPDX-License-Identifier: Apache-2.0
// GreenAttention A3: Gated-DeltaNet replay-SSM *verify* kernels for SM90 (H100).
//
// Drop-in for `gdn_replayssm_spec_circular_kernel` with IS_FLUSH=False, fp32 or bf16
// checkpoint, `beta_cache=None` (see docs/gdn-contract.md section 4; ported semantics from
// SGLang's kernels/ops/attention/linear/gdn_replayssm_spec_decode.py, Apache-2.0, itself derived
// from ReplaySSM / vLLM).
//
// The checkpoint state S0 [V=128, K=128] is read exactly once and never written: all of the
// verify math is expressed through
//     S0 . q'_j, S0 . k'_j                         (8 vectors, 8 dot products per state row)
// plus a rank-<=32 history correction from the replay ring and a 4x4 triangular solve.
//
// Two kernels, launched back to back on one stream (the second with programmatic dependent
// launch so its state loads overlap the first):
//
//   verify_prep_kernel  grid (B, H)   -- no state.  l2 norms, Gram entries, q'/k' vectors, the
//                       ring write of (k', residual), history scores khist_c . vec_m, per-value-
//                       head decay, gating, (I+A)^-1 and the 4x4 F matrix.  Everything that does
//                       not touch S0.  Results go to a persistent workspace.
//   verify_kernel       grid (B, HV)  -- thread t owns state row v = t.  Issues all 64 KB of
//                       state loads, then the 8 dots per row (butterfly reduce-scatter over the
//                       8 lanes sharing a row), the history accumulate and a short scalar tail
//                       (D, O and the output / ring stores).
//
// Why split: with the previous single kernel the serialized per-CTA post-load phase (norm/Gram
// reduce, history scores, redundant gating tail) cost ~11 us with only 2 CTAs/SM, so the kernel was
// latency-bound (~1.0x Triton) although the state stream alone runs at ~3 TB/s.
//
// Math is fp32 FMA on CUDA cores (arithmetic is <10% of the FP32 peak at the HBM roofline),
// i.e. it does not use TF32 and is more accurate than the Triton incumbent.
#pragma once

#include "sm90/gdn_recurrent/kernel.cuh"

namespace green {
namespace gdn {

constexpr int kMaxSpec = 4;   // draft tokens verified per call
constexpr int kMaxRing = 32;  // replay ring length (power of two, <= 32)

// Per-(sequence, value-head) scalar block written by the prep kernel (32 floats):
//   [0..5]   Tm[i][j], i > j, at i*(i-1)/2 + j     (T = (I + A)^-1, strictly lower part)
//   [6..9]   beta_j      [10..13] exp(G_j)      [14..17] g_j
//   [18..27] F[i][j], i <= j, at 18 + j*(j+1)/2 + i
//   [28]     total_decay
constexpr int kScalars = 32;
constexpr int kWsVecPerSeq = kH * 8 * kK;       // floats of ws_vec per sequence
constexpr int kWsCoefPerSeq = kHV * kMaxRing * 8;
constexpr int kWsScPerSeq = kHV * kScalars;

struct VerifyParams {
  const __nv_bfloat16* q;  // token 0, head 0 (q/k head stride = K, v/out head stride = V)
  const __nv_bfloat16* k;
  const __nv_bfloat16* v;
  int64_t q_ts, k_ts, v_ts;
  const __nv_bfloat16* a;
  const __nv_bfloat16* b;
  int64_t a_ts, b_ts;
  const float* A_log;
  const void* dt_bias;
  int dt_bias_bf16;
  const void* state;  // [slots, HV, V, K], ST
  int64_t state_ss;
  __nv_bfloat16* d_cache;  // [ring, HV, L, V]
  __nv_bfloat16* rawv_cache;
  __nv_bfloat16* k_cache;  // [ring, H, L, K]
  __nv_bfloat16* rawk_cache;
  float* g_cache;  // [ring, HV, L]
  int64_t d_ss, rawv_ss, k_ss, rawk_ss, g_ss;
  const void* state_idx;   // [B] int32|int64
  const void* replay_idx;  // [B] int32|int64
  int sidx64, ridx64;
  const int* qsl;  // [B+1]
  const int* write_pos;
  const int* cache_base;
  const int8_t* is_flush;
  __nv_bfloat16* out;  // [T, HV, V]
  int64_t o_ts;
  float scale;
  int L;  // ring length
  float* ws_vec;   // [maxB, H, 8, K]   q'_0..3, k'_0..3 in vpos<VEC> order
  float* ws_coef;  // [maxB, HV, 32, 8] decay_c * (khist_c . vec_m)
  float* ws_sc;    // [maxB, HV, 32]    see kScalars
};

// stats[] slot layout (raw, un-normalised products reduced over K):
//   0..3   |q_j|^2                4..7   |k_j|^2
//   8..13  k_i.k_j  for i>j      14..23 k_i.q_j for i<=j
__host__ __device__ constexpr int kk_slot(int i, int j) {  // i > j
  return 8 + (i == 1 ? 0 : i == 2 ? 1 + j : 3 + j);
}
__host__ __device__ constexpr int kq_slot(int i, int j) {  // i <= j
  return 14 + (i == 0 ? j : i == 1 ? 3 + j : i == 2 ? 5 + j : 6 + j);
}

// Position of state column `col` inside a vecs[] row (chosen so that every vector read of the
// state-dot loop is a conflict-free LDS.128 for both state dtypes).
template <int VEC>
__device__ __forceinline__ int vpos(int col) {
  if constexpr (VEC == 4) {
    return col;
  } else {
    const int w = col & 7;
    return (w >> 2) * 64 + (col >> 3) * 4 + (w & 3);
  }
}
// Index of the float4 holding half `hh` of the 8-column chunk `chunk`.
template <int VEC>
__device__ __forceinline__ int vq4(int chunk, int hh) {
  if constexpr (VEC == 4) {
    return chunk * 8 + hh * 4;
  } else {
    return hh * 64 + chunk * 4;
  }
}

template <typename ST>
struct RawTile;
template <>
struct RawTile<float> {
  using T = float4;
  __device__ __forceinline__ static T load(const float* p) { return ld16(p); }
  __device__ __forceinline__ static void unpack(const T& r, float (&x)[4]) {
    x[0] = r.x; x[1] = r.y; x[2] = r.z; x[3] = r.w;
  }
};
template <>
struct RawTile<__nv_bfloat16> {
  using T = uint4;
  __device__ __forceinline__ static T load(const __nv_bfloat16* p) { return ld16u(p); }
  __device__ __forceinline__ static void unpack(const T& r, float (&x)[8]) {
    const uint32_t w[4] = {r.x, r.y, r.z, r.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      x[2 * i] = __uint_as_float(w[i] << 16);
      x[2 * i + 1] = __uint_as_float(w[i] & 0xffff0000u);
    }
  }
};

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
  const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
__device__ __forceinline__ void cp_async_wait_all() { asm volatile("cp.async.wait_all;\n" ::); }

// Programmatic dependent launch (no-ops when the kernel was launched without the attribute).
__device__ __forceinline__ void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;\n" ::); }
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;\n" ::: "memory"); }

// 32-lane butterfly reduce-scatter of 32 values: lane c ends with the full sum of slot c.
__device__ __forceinline__ float reduce_scatter32(float (&v)[32], int lane) {
#pragma unroll
  for (int s = 4; s >= 0; --s) {
    const int half = 1 << s;
    const bool hi = (lane >> s) & 1;
#pragma unroll
    for (int i = 0; i < half; ++i) {
      const float send = hi ? v[i] : v[i + half];
      const float keep = hi ? v[i + half] : v[i];
      v[i] = keep + __shfl_xor_sync(kFull, send, half);
    }
  }
  return v[0];
}

__device__ __forceinline__ int64_t load_idx(const void* p, int is64, int n) {
  return is64 ? static_cast<const int64_t*>(p)[n] : static_cast<int64_t>(static_cast<const int*>(p)[n]);
}

// ---------------------------------------------------------------------------------------------
// prep kernel: one CTA per (sequence, key head); 128 threads, thread t = column t of q/k.
// ---------------------------------------------------------------------------------------------
template <typename ST>
__global__ void __launch_bounds__(128) verify_prep_kernel(const VerifyParams p) {
  constexpr int VEC = StateVec<ST>::VEC;

  __shared__ __align__(16) float vecs[8][kK];
  __shared__ __align__(16) float scs[kMaxRing][8];  // raw scores khist_c . vec_m (decay applied later)
  __shared__ float red[4][32];
  __shared__ __align__(16) float stats[32];

  const int t = threadIdx.x;
  const int lane = t & 31;
  const int w = t >> 5;
  const int n = blockIdx.x / kH;
  const int h = blockIdx.x % kH;

  pdl_trigger();  // the state-streaming kernel may start issuing its loads now

  const int64_t sidx = load_idx(p.state_idx, p.sidx64, n);
  const int bos = p.qsl[n];
  const int nt = min(p.qsl[n + 1] - bos, kMaxSpec);
  if (nt <= 0 || sidx < 0) return;  // main kernel zero-fills NULL rows
  const int64_t r = load_idx(p.replay_idx, p.ridx64, n);
  if (p.is_flush[r] != 0) return;   // the flush kernel owns this row
  const int L = p.L;
  const int Lm = L - 1;
  const int base = p.cache_base[r];
  const int wp = p.write_pos[r];
  const int W = min(max(wp, 0), L);

  float x[kMaxSpec], y[kMaxSpec];
#pragma unroll
  for (int j = 0; j < kMaxSpec; ++j) {
    const bool on = j < nt;  // masked tokens are zero
    x[j] = on ? bf2f(p.q[(bos + j) * p.q_ts + h * kK + t]) : 0.0f;
    y[j] = on ? bf2f(p.k[(bos + j) * p.k_ts + h * kK + t]) : 0.0f;
  }
  const int hrow = t >> 2;  // history row handled by this thread (4 threads per row)
  const int hsub = t & 3;
  uint4 kraw[4];
  float gh = 0.0f;  // warps 0..2: g history of value head 3h+w, lane = ring offset
  if (W > 0) {
    if (hrow < W) {
      const __nv_bfloat16* ksrc =
          p.k_cache + r * p.k_ss + (static_cast<int64_t>(h) * L + ((base + hrow) & Lm)) * kK;
#pragma unroll
      for (int i = 0; i < 4; ++i) kraw[i] = ld16u(ksrc + (hsub + 4 * i) * 8);
    }
    if (w < kHV / kH && lane < W)
      gh = p.g_cache[r * p.g_ss + static_cast<int64_t>(h * (kHV / kH) + w) * L + ((base + lane) & Lm)];
  }

  // ---- l2 norms and the kk / kq Gram entries from one fused block reduction --------------------
  {
    float val[32];
#pragma unroll
    for (int s = 0; s < 32; ++s) val[s] = 0.0f;
#pragma unroll
    for (int j = 0; j < kMaxSpec; ++j) {
      val[j] = x[j] * x[j];
      val[4 + j] = y[j] * y[j];
#pragma unroll
      for (int i = 0; i < kMaxSpec; ++i) {
        if (i > j) val[kk_slot(i, j)] = y[i] * y[j];
        if (i <= j) val[kq_slot(i, j)] = y[i] * x[j];
      }
    }
    red[w][lane] = reduce_scatter32(val, lane);
  }
  __syncthreads();
  if (t < 24) stats[t] = red[0][t] + red[1][t] + red[2][t] + red[3][t];
  __syncthreads();

  float qs[kMaxSpec], kr[kMaxSpec];  // q' scale (incl. softmax scale) and k' scale per token
#pragma unroll
  for (int j = 0; j < kMaxSpec; ++j) {
    qs[j] = (1.0f / sqrtf(stats[j] + 1e-6f)) * p.scale;
    kr[j] = 1.0f / sqrtf(stats[4 + j] + 1e-6f);
  }
  float* wsv = p.ws_vec + static_cast<int64_t>(n * kH + h) * (8 * kK);
#pragma unroll
  for (int j = 0; j < kMaxSpec; ++j) {
    const float qv = x[j] * qs[j];
    const float kv = y[j] * kr[j];
    vecs[j][vpos<VEC>(t)] = qv;
    vecs[4 + j][vpos<VEC>(t)] = kv;
    wsv[j * kK + vpos<VEC>(t)] = qv;
    wsv[(4 + j) * kK + vpos<VEC>(t)] = kv;
    // ring write of the (hi, residual) bf16 pair of k'
    if (j < nt && wp + j < L) {
      const int64_t phys = (base + wp + j) & Lm;
      const __nv_bfloat16 hi = __float2bfloat16_rn(kv);
      const __nv_bfloat16 lo = __float2bfloat16_rn(kv - bf2f(hi));
      p.k_cache[r * p.k_ss + (static_cast<int64_t>(h) * L + phys) * kK + t] = hi;
      p.rawk_cache[r * p.rawk_ss + (static_cast<int64_t>(h) * L + phys) * kK + t] = lo;
    }
  }
  __syncthreads();

  // ---- history scores khist_c . vec_m for the key head --------------------------------------
  if (W > 0) {
    float sc[8];
#pragma unroll
    for (int m = 0; m < 8; ++m) sc[m] = 0.0f;
    if (hrow < W) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        float kx[8];
        RawTile<__nv_bfloat16>::unpack(kraw[i], kx);
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
#pragma unroll
          for (int m = 0; m < 8; ++m) {
            const float4 vv = *reinterpret_cast<const float4*>(&vecs[m][vq4<VEC>(hsub + 4 * i, hh)]);
            sc[m] += kx[hh * 4 + 0] * vv.x + kx[hh * 4 + 1] * vv.y + kx[hh * 4 + 2] * vv.z + kx[hh * 4 + 3] * vv.w;
          }
        }
      }
    }
#pragma unroll
    for (int m = 0; m < 8; ++m) {
      sc[m] += __shfl_xor_sync(kFull, sc[m], 1);
      sc[m] += __shfl_xor_sync(kFull, sc[m], 2);
    }
    if (hsub == 0 && hrow < W) {
      *reinterpret_cast<float4*>(&scs[hrow][0]) = make_float4(sc[0], sc[1], sc[2], sc[3]);
      *reinterpret_cast<float4*>(&scs[hrow][4]) = make_float4(sc[4], sc[5], sc[6], sc[7]);
    }
  }
  __syncthreads();

  // ---- per value head (warps 0..2): decay, gating, T = (I+A)^-1, F ------------------------------
  if (w >= kHV / kH) return;
  const int hv = h * (kHV / kH) + w;
  float total_decay = 1.0f;
  if (W > 0) {
    float pre = gh;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const float u = __shfl_up_sync(kFull, pre, o);
      if (lane >= o) pre += u;
    }
    const float total = __shfl_sync(kFull, pre, 31);
    total_decay = expf(total);
    if (lane < W) {
      const float dec = expf(total - pre);
      float* cdst = p.ws_coef + (static_cast<int64_t>(n) * kHV + hv) * (kMaxRing * 8) + lane * 8;
      *reinterpret_cast<float4*>(cdst) =
          make_float4(dec * scs[lane][0], dec * scs[lane][1], dec * scs[lane][2], dec * scs[lane][3]);
      *reinterpret_cast<float4*>(cdst + 4) =
          make_float4(dec * scs[lane][4], dec * scs[lane][5], dec * scs[lane][6], dec * scs[lane][7]);
    }
  }

  const float dtb = ldf(p.dt_bias, p.dt_bias_bf16, hv);
  const float Alog = p.A_log[hv];
  float g[kMaxSpec], beta[kMaxSpec], G[kMaxSpec], eG[kMaxSpec];
  float run = 0.0f;
#pragma unroll
  for (int j = 0; j < kMaxSpec; ++j) {
    g[j] = 0.0f;
    beta[j] = 0.0f;
    if (j < nt) {
      g[j] = softplus_gate(bf2f(p.a[(bos + j) * p.a_ts + hv]), dtb, Alog);
      beta[j] = 1.0f / (1.0f + expf(-bf2f(p.b[(bos + j) * p.b_ts + hv])));
    }
    run += g[j];
    G[j] = run;
    eG[j] = expf(run);
  }

  float Tm[kMaxSpec][kMaxSpec];  // T = (I + A)^-1, lower triangular
#pragma unroll
  for (int i = 0; i < kMaxSpec; ++i)
#pragma unroll
    for (int j = 0; j < kMaxSpec; ++j) Tm[i][j] = (i == j) ? 1.0f : 0.0f;
#pragma unroll
  for (int i = 1; i < kMaxSpec; ++i) {
#pragma unroll
    for (int j = 0; j < i; ++j) {
      float s = 0.0f;
#pragma unroll
      for (int m = j; m < i; ++m) {
        const float Aim = beta[i] * expf(G[i] - G[m]) * (stats[kk_slot(i, m)] * kr[i] * kr[m]);
        s = fmaf(Aim, Tm[m][j], s);
      }
      Tm[i][j] = -s;
    }
  }

  if (lane == 0) {
    float s[kScalars];
#pragma unroll
    for (int i = 0; i < kScalars; ++i) s[i] = 0.0f;
#pragma unroll
    for (int i = 1; i < kMaxSpec; ++i)
#pragma unroll
      for (int j = 0; j < i; ++j) s[i * (i - 1) / 2 + j] = Tm[i][j];
#pragma unroll
    for (int j = 0; j < kMaxSpec; ++j) {
      s[6 + j] = beta[j];
      s[10 + j] = eG[j];
      s[14 + j] = g[j];
#pragma unroll
      for (int i = 0; i <= j; ++i)
        s[18 + j * (j + 1) / 2 + i] = expf(G[j] - G[i]) * (stats[kq_slot(i, j)] * kr[i] * qs[j]);
    }
    s[28] = total_decay;
    float4* dst = reinterpret_cast<float4*>(p.ws_sc + (static_cast<int64_t>(n) * kHV + hv) * kScalars);
#pragma unroll
    for (int i = 0; i < kScalars / 4; ++i) dst[i] = make_float4(s[4 * i], s[4 * i + 1], s[4 * i + 2], s[4 * i + 3]);
  }
}

// ---------------------------------------------------------------------------------------------
// main kernel: one CTA per (sequence, value head); 128 threads, thread t = state row t.
// ---------------------------------------------------------------------------------------------
template <typename ST>
__global__ void __launch_bounds__(128, 2) verify_kernel(const VerifyParams p) {
  using Tile = RawTile<ST>;
  constexpr int VEC = StateVec<ST>::VEC;  // state columns per 16 B
  constexpr int NI = 16 / VEC;            // column slices per row per lane (8 lanes x VEC x NI = K)

  __shared__ __align__(16) float vecs[8][kK];                  // q'_0..3, k'_0..3
  __shared__ __align__(16) __nv_bfloat16 dtile[kMaxRing][kV];  // d history, row = ring offset
  __shared__ __align__(16) float coef[kMaxRing][8];            // decay_c * (khist_c . vec_m)
  __shared__ __align__(16) float scs[kScalars];

  const int t = threadIdx.x;
  const int lane = t & 31;
  const int w = t >> 5;
  const int n = blockIdx.x / kHV;
  const int hv = blockIdx.x % kHV;
  const int h = hv / (kHV / kH);

  // ---- uniform metadata and early exits -------------------------------------------------
  const int64_t sidx = load_idx(p.state_idx, p.sidx64, n);
  const int bos = p.qsl[n];
  const int nt = min(p.qsl[n + 1] - bos, kMaxSpec);
  if (nt <= 0) return;
  if (sidx < 0) {  // NULL block: zero the outputs, nothing else
    for (int j = 0; j < nt; ++j)
      p.out[(bos + j) * p.o_ts + hv * kV + t] = __float2bfloat16_rn(0.0f);
    return;
  }
  const int64_t r = load_idx(p.replay_idx, p.ridx64, n);
  if (p.is_flush[r] != 0) return;  // the flush kernel owns this row
  const int L = p.L;
  const int Lm = L - 1;
  const int base = p.cache_base[r];
  const int wp = p.write_pos[r];
  const int W = min(max(wp, 0), L);  // valid history rows

  // The workspace written by the prep kernel is consumed first: the grid-dependency wait
  // (griddepcontrol.wait) returns once the prep grid has completed and flushed its memory. The
  // two grids still overlap through the launch/ramp-up of this one.
  pdl_wait();
  {
    const float* vsrc = p.ws_vec + static_cast<int64_t>(n * kH + h) * (8 * kK);
    for (int i = t; i < 8 * kK / 4; i += 128) cp_async16(&vecs[0][0] + 4 * i, vsrc + 4 * i);
    if (W > 0) {
      const float* csrc = p.ws_coef + (static_cast<int64_t>(n) * kHV + hv) * (kMaxRing * 8);
      for (int i = t; i < W * 2; i += 128) cp_async16(&coef[0][0] + 4 * i, csrc + 4 * i);
    }
    if (t < kScalars / 4)
      cp_async16(&scs[4 * t], p.ws_sc + (static_cast<int64_t>(n) * kHV + hv) * kScalars + 4 * t);
  }
  // ---- issue the remaining long-latency loads: d history tile, v, and the state ----------------
  if (W > 0) {  // d history tile -> smem (cp.async, 16 B)
    const __nv_bfloat16* dsrc = p.d_cache + r * p.d_ss + static_cast<int64_t>(hv) * L * kV;
    for (int idx = t; idx < W * (kV / 8); idx += 128) {
      const int c = idx >> 4;
      const int ch = idx & 15;
      cp_async16(&dtile[c][ch * 8], dsrc + static_cast<int64_t>((base + c) & Lm) * kV + ch * 8);
    }
  }
  float vj[kMaxSpec];
#pragma unroll
  for (int j = 0; j < kMaxSpec; ++j) vj[j] = j < nt ? bf2f(p.v[(bos + j) * p.v_ts + hv * kV + t]) : 0.0f;

  typename Tile::T raw[8][NI];
  {
    const ST* sp = static_cast<const ST*>(p.state) + sidx * p.state_ss +
                   static_cast<int64_t>(hv * kV + w * 32 + (lane >> 3) * 8) * kK + (lane & 7) * VEC;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
#pragma unroll
      for (int i = 0; i < NI; ++i) raw[j][i] = Tile::load(sp + static_cast<int64_t>(j) * kK + i * 8 * VEC);
    }
  }

  // ---- everything issued above lands here -------------------------------------------------------
  cp_async_wait_all();
  __syncthreads();

  // ---- state dots: hw[m] = S0[v] . vec_m, two passes of 4 vectors (q' first, then k') ------------
  float hw[8];
#pragma unroll
  for (int pass = 0; pass < 2; ++pass) {
    float acc[8][4];
#pragma unroll
    for (int j = 0; j < 8; ++j)
#pragma unroll
      for (int mm = 0; mm < 4; ++mm) acc[j][mm] = 0.0f;
#pragma unroll
    for (int i = 0; i < NI; ++i) {
#pragma unroll
      for (int hh = 0; hh < VEC / 4; ++hh) {
        float4 vv[4];
#pragma unroll
        for (int mm = 0; mm < 4; ++mm)
          vv[mm] = *reinterpret_cast<const float4*>(&vecs[4 * pass + mm][hh * 64 + (8 * i + (lane & 7)) * 4]);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          float sv[VEC];
          Tile::unpack(raw[j][i], sv);
#pragma unroll
          for (int mm = 0; mm < 4; ++mm) {
            acc[j][mm] = fmaf(sv[hh * 4 + 0], vv[mm].x, acc[j][mm]);
            acc[j][mm] = fmaf(sv[hh * 4 + 1], vv[mm].y, acc[j][mm]);
            acc[j][mm] = fmaf(sv[hh * 4 + 2], vv[mm].z, acc[j][mm]);
            acc[j][mm] = fmaf(sv[hh * 4 + 3], vv[mm].w, acc[j][mm]);
          }
        }
      }
    }
    // reduce over the 8 lanes of the row group; lane c keeps row c
    const int c8 = lane & 7;
#pragma unroll
    for (int s = 2; s >= 0; --s) {
      const int half = 1 << s;
      const bool hi = (c8 >> s) & 1;
#pragma unroll
      for (int i = 0; i < half; ++i) {
#pragma unroll
        for (int mm = 0; mm < 4; ++mm) {
          const float send = hi ? acc[i][mm] : acc[i + half][mm];
          const float keep = hi ? acc[i + half][mm] : acc[i][mm];
          acc[i][mm] = keep + __shfl_xor_sync(kFull, send, half);
        }
      }
    }
#pragma unroll
    for (int mm = 0; mm < 4; ++mm) hw[4 * pass + mm] = acc[0][mm];
  }

  // ---- history correction: hw = total_decay * hw + sum_c d[c, v] * coef[c, :] -------------------
  if (W > 0) {
    const float total_decay = scs[28];
#pragma unroll
    for (int m = 0; m < 8; ++m) hw[m] *= total_decay;
    for (int c = 0; c < W; ++c) {
      const float dd = bf2f(dtile[c][t]);
      const float4 c0 = *reinterpret_cast<const float4*>(&coef[c][0]);
      const float4 c1 = *reinterpret_cast<const float4*>(&coef[c][4]);
      hw[0] = fmaf(dd, c0.x, hw[0]); hw[1] = fmaf(dd, c0.y, hw[1]);
      hw[2] = fmaf(dd, c0.z, hw[2]); hw[3] = fmaf(dd, c0.w, hw[3]);
      hw[4] = fmaf(dd, c1.x, hw[4]); hw[5] = fmaf(dd, c1.y, hw[5]);
      hw[6] = fmaf(dd, c1.z, hw[6]); hw[7] = fmaf(dd, c1.w, hw[7]);
    }
  }

  // ---- tail: D and O for the spec tokens (T, beta, exp(G), F come precomputed) ----------------
  float Rj[kMaxSpec], D[kMaxSpec];
#pragma unroll
  for (int j = 0; j < kMaxSpec; ++j) Rj[j] = scs[6 + j] * (vj[j] - scs[10 + j] * hw[4 + j]);
#pragma unroll
  for (int i = 0; i < kMaxSpec; ++i) {
    float s = Rj[i];  // T[i][i] = 1
#pragma unroll
    for (int j = 0; j < i; ++j) s = fmaf(Rj[j], scs[i * (i - 1) / 2 + j], s);
    D[i] = s;
  }

#pragma unroll
  for (int j = 0; j < kMaxSpec; ++j) {
    if (j >= nt) break;
    float o = scs[10 + j] * hw[j];
#pragma unroll
    for (int i = 0; i <= j; ++i) o = fmaf(D[i], scs[18 + j * (j + 1) / 2 + i], o);
    p.out[(bos + j) * p.o_ts + hv * kV + t] = __float2bfloat16_rn(o);
    if (wp + j < L) {
      const int64_t phys = (base + wp + j) & Lm;
      const __nv_bfloat16 hi = __float2bfloat16_rn(D[j]);
      const __nv_bfloat16 lo = __float2bfloat16_rn(D[j] - bf2f(hi));
      p.d_cache[r * p.d_ss + (static_cast<int64_t>(hv) * L + phys) * kV + t] = hi;
      p.rawv_cache[r * p.rawv_ss + (static_cast<int64_t>(hv) * L + phys) * kV + t] = lo;
      if (t == 0) p.g_cache[r * p.g_ss + static_cast<int64_t>(hv) * L + phys] = scs[14 + j];
    }
  }
}

}  // namespace gdn
}  // namespace green
