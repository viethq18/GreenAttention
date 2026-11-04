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
//   verify_kernel       persistent, min(B*HV, #SMs) CTAs of 160 threads, tiles (seq, hv) strided
//                       over CTAs.  Warp 4 is the producer: bulk TMA of the 64 KB (fp32) state
//                       tile and cp.async of the aux block (prep vectors, history d rows, coefs,
//                       scalars, v rows) into one of NST=2 smem stages.  Warps 0-3 consume: thread
//                       t owns state row v = t, 8 dots per row from smem, history accumulate,
//                       scalar tail (D, O and the output / ring stores).  The state TMA of the
//                       first NST tiles is issued before the PDL wait, so it overlaps prep.
//
// Why split + persistent: the post-load phase that does not touch S0 (norm/Gram reduce, history
// scores, gating) cost ~11 us serialized inside each CTA, and a one-CTA-per-tile grid at 2 CTAs/SM
// serialized load -> math -> store.  The producer warp now streams tile i+NST while the consumers
// compute tile i.
//
// Math is fp32 FMA on CUDA cores (arithmetic is <10% of the FP32 peak at the HBM roofline),
// i.e. it does not use TF32 and is more accurate than the Triton incumbent.
#pragma once

#include "common/ptx.cuh"
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

// Dynamic smem of the persistent verify kernel: NST stages of (state tile + aux) followed by
// NST Info records (32 B) and 2*NST mbarriers.  aux = vecs[8][K] f32 | dtile[32][V] bf16 |
// coef[32][8] f32 | scs[32] f32 | vtile[4][V] bf16.
constexpr int kVerifyAuxBytes = 8 * kK * 4 + kMaxRing * kV * 2 + kMaxRing * 8 * 4 + kScalars * 4 +
                                kMaxSpec * kV * 2;
template <typename ST, int NST>
constexpr int verify_smem_bytes() {
  return NST * (kV * kK * static_cast<int>(sizeof(ST)) + kVerifyAuxBytes) + NST * (32 + 16);
}

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

// Register select that LLVM cannot fold into a select of *addresses* (which demotes the whole
// v[] array to local memory: 66 LDL/STL in the prep kernel before this).
__device__ __forceinline__ float sel_f32(bool c, float a, float b) {
  float r;
  asm("{\n\t.reg .pred p;\n\tsetp.ne.u32 p, %3, 0;\n\tselp.f32 %0, %1, %2, p;\n\t}"
      : "=f"(r) : "f"(a), "f"(b), "r"(static_cast<unsigned>(c)));
  return r;
}

// 32-lane butterfly reduce-scatter of 32 values: lane c ends with the full sum of slot c.
__device__ __forceinline__ float reduce_scatter32(float (&v)[32], int lane) {
#pragma unroll
  for (int s = 4; s >= 0; --s) {
    const int half = 1 << s;
    const bool hi = (lane >> s) & 1;
#pragma unroll
    for (int i = 0; i < half; ++i) {
      const float a = v[i], b = v[i + half];
      const float send = sel_f32(hi, a, b);
      const float keep = sel_f32(hi, b, a);
      v[i] = keep + __shfl_xor_sync(kFull, send, half);
    }
  }
  return v[0];
}

__device__ __forceinline__ int64_t load_idx(const void* p, int is64, int n) {
  return is64 ? static_cast<const int64_t*>(p)[n] : static_cast<int64_t>(static_cast<const int*>(p)[n]);
}

// cp.async.bulk 1D g2s with the completion tracked by an mbarrier transaction count.
__device__ __forceinline__ void expect_tx(uint32_t bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(bar),
               "r"(bytes) : "memory");
}
__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
      ::"r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
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
  // gate inputs of value head 3h+w (warps 0..2): issued with the other loads, not behind the
  // block reductions
  float araw[kMaxSpec], braw[kMaxSpec], dtb = 0.0f, Alog = 0.0f;
  if (w < kHV / kH) {
    const int hv0 = h * (kHV / kH) + w;
#pragma unroll
    for (int j = 0; j < kMaxSpec; ++j) {
      araw[j] = 0.0f;
      braw[j] = 0.0f;
      if (j < nt) {
        araw[j] = bf2f(p.a[(bos + j) * p.a_ts + hv0]);
        braw[j] = bf2f(p.b[(bos + j) * p.b_ts + hv0]);
      }
    }
    dtb = ldf(p.dt_bias, p.dt_bias_bf16, hv0);
    Alog = p.A_log[hv0];
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

  float g[kMaxSpec], beta[kMaxSpec], G[kMaxSpec], eG[kMaxSpec];
  float run = 0.0f;
#pragma unroll
  for (int j = 0; j < kMaxSpec; ++j) {
    g[j] = 0.0f;
    beta[j] = 0.0f;
    if (j < nt) {
      g[j] = softplus_gate(araw[j], dtb, Alog);
      beta[j] = 1.0f / (1.0f + expf(-braw[j]));
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
// main kernel: persistent, warp-specialized.  min(B*HV, numSMs) CTAs, one per SM; tile
// g = (sequence n, value head hv) -> n = g / kHV, hv = g % kHV, round-robin g = blockIdx + i*G.
// Warp 4 is the producer: a 64 KB cp.async.bulk 1D copy of the state tile (fp32) or 4 x 16 KB
// chunks (bf16) plus cp.async of every other input; warps 0..3 are consumers, thread t =
// state row t, NST=2 stages in dynamic smem.  One mbarrier pair per stage: full[s] (count 33:
// lane 0 expect_tx arrive + 32 cp.async arrives), empty[s] (128 consumer arrives).  A stage
// (state tile + aux) is released to the producer while the next tile's loads are in flight,
// so per-CTA compute overlaps the state stream of the next tile instead of serializing.
//
// Why persistent: at B=32 the old tile-per-CTA kernel ran ~6 waves of 2 CTAs/SM, each CTA
// serializing load -> compute -> store (~15 us of compute exposed per kernel) and exposing
// the prep latency of every wave behind its pdl_wait.  With one resident CTA per SM and a
// 2-stage TMA pipeline the state stream never drains between tiles and the exposed prep
// shrinks to the producer's single pdl_wait for its first tile.
// ---------------------------------------------------------------------------------------------
template <typename ST, int NST>
__global__ void __launch_bounds__(160, 1) verify_kernel(const VerifyParams p, int ntiles, int nseq) {
  using Tile = RawTile<ST>;
  constexpr int VEC = StateVec<ST>::VEC;   // state columns per 16 B load
  constexpr int NI = 16 / VEC;             // column slices per row per lane (8 lanes x VEC x NI = K)

  // stage layout (bytes): state tile | vecs[8][K] | dtile[32][V] | coef[32][8] | scs[32] |
  // vtile[4][V]   (all bf16/f32 blocks 16 B aligned; the float offsets below are in floats)
  constexpr int SB = kV * kK * static_cast<int>(sizeof(ST));       // state bytes
  constexpr int VEC_B = 8 * kK * 4;        // vecs bytes
  constexpr int DT_B = kMaxRing * kV * 2;  // dtile bytes
  constexpr int CO_B = kMaxRing * 8 * 4;   // coef bytes
  constexpr int SC_B = kScalars * 4;       // scs bytes
  constexpr int AUX = VEC_B + DT_B + CO_B + SC_B + kMaxSpec * kV * 2;
  static_assert(AUX == kVerifyAuxBytes, "launcher smem size out of sync");
  constexpr int O_D = SB + VEC_B;                  // dtile byte offset in stage (after vecs)
  constexpr int O_C = O_D + DT_B;                  // coef
  constexpr int O_S = O_C + CO_B;                  // scs
  constexpr int O_V = O_S + SC_B;                  // vtile

  extern __shared__ __align__(128) char smem_raw[];
  struct Info {  // written by producer lane 0 before its arrive on full[s]
    int mode;    // 0 skip, 1 compute, 2 zero-fill
    int nt, bos, r, wp, base, W, Lm;
  };
  char* tail = smem_raw + NST * (SB + AUX);
  Info* info = reinterpret_cast<Info*>(tail);
  static_assert(sizeof(Info) == 32, "launcher smem size out of sync");
  uint64_t* bars = reinterpret_cast<uint64_t*>(tail + NST * sizeof(Info));

  const int t = threadIdx.x;
  const int lane = t & 31;
  const int w = t >> 5;

  if (t < 2 * NST) mbar_init(smem_u32(&bars[t]), (t & 1) ? 128u : 33u);
  if (t == 0) asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  __syncthreads();
  uint32_t full[NST], empty[NST];
#pragma unroll
  for (int s = 0; s < NST; ++s) {
    full[s] = smem_u32(&bars[2 * s]);
    empty[s] = smem_u32(&bars[2 * s + 1]);
  }

  const int G = gridDim.x;

  if (w == 4) {
    // ---- producer: lane j holds the metadata of this CTA's tile i0+j (tile g = blockIdx + i*G;
    // up to 32 tiles per refill, broadcast by shuffle so nothing is indexed dynamically), then
    // per-tile loads ---------------------------------------------------------------------------
    constexpr int kMB = 32;
    int i0 = -1000000;
    int m_bos = 0, m_nt = 0, m_wp = 0, m_cb = 0, m_fl = 1;
    int64_t m_sd = -1, m_ri = -1;
    const int L = p.L, Lm = L - 1;

    auto ld_meta = [&](int ii) {
      i0 = ii;
      const int gg = blockIdx.x + (ii + lane) * G;
      const int nn = max(0, min(gg / kHV, nseq - 1));  // this CTA's tile ii+lane -> its sequence
      m_sd = load_idx(p.state_idx, p.sidx64, nn);
      m_ri = load_idx(p.replay_idx, p.ridx64, nn);
      const int b1 = p.qsl[nn], b2 = p.qsl[nn + 1];
      int cb = 0, wp = 0, fl = 1;
      if (m_ri >= 0) { cb = p.cache_base[m_ri]; wp = p.write_pos[m_ri]; fl = p.is_flush[m_ri]; }
      m_bos = b1; m_nt = min(b2 - b1, kMaxSpec); m_wp = min(max(wp, 0), L);
      m_cb = cb; m_fl = fl;
    };

    auto st_ = [&](int s) { return smem_raw + s * (SB + AUX); };

    // Barrier accounting of a computed tile: lane 0 expect_tx arrive (1) + one noinc arrive per
    // lane (32) = 33 = init count.  The bulk state TMA of the first NST tiles is issued before the
    // pdl_wait (the checkpoint state is independent of prep); every cp.async that reads prep
    // output goes strictly after it.
    auto issue_state = [&](int hv, int s, int64_t sdv) {
      const uint32_t dst = smem_u32(st_(s));
      const ST* sbase = static_cast<const ST*>(p.state) + sdv * p.state_ss +
                        static_cast<int64_t>(hv * kV) * kK;
      if (lane == 0) expect_tx(full[s], SB);
      if (sizeof(ST) == 4) {
        if (lane == 0) bulk_g2s(dst, sbase, SB, full[s]);
      } else if (lane < 4) {  // bf16 tile: 4 x 16 KB chunks
        bulk_g2s(dst + lane * (SB / 4), sbase + lane * (kV * kK / 4), SB / 4, full[s]);
      }
    };
    auto issue_aux = [&](int hv, int s, int n, int64_t riv, int bosn, int ntn, int W, int cbv) {
      char* stg = st_(s);
      if (W > 0) {
        const __nv_bfloat16* dsrc = p.d_cache + riv * p.d_ss + static_cast<int64_t>(hv) * L * kV;
        const uint32_t ddst = smem_u32(stg + O_D);
        for (int idx = lane; idx < W * (kV / 8); idx += 32) {
          const int c = idx >> 4, ch = idx & 15;
          cp_async_16(ddst + (c * kV + ch * 8) * 2,
                      dsrc + static_cast<int64_t>((cbv + c) & Lm) * kV + ch * 8);
        }
      }
      const uint32_t vdst = smem_u32(stg + O_V);
      for (int jj = lane; jj < kMaxSpec * (kV / 8); jj += 32) {
        const int j = jj >> 4, ch = jj & 15;  // rows j >= nt re-read the last valid row (beta = 0)
        cp_async_16(vdst + (j * kV + ch * 8) * 2,
                    p.v + static_cast<int64_t>(bosn + min(j, ntn - 1)) * p.v_ts + hv * kV + ch * 8);
      }
      const int h = hv / (kHV / kH);
      const float* vsrc = p.ws_vec + static_cast<int64_t>(n * kH + h) * (8 * kK);
      const uint32_t adst = smem_u32(stg + SB);
      for (int i = lane; i < VEC_B / 16; i += 32) cp_async_16(adst + 16 * i, vsrc + 4 * i);
      if (W > 0) {
        const float* csrc = p.ws_coef + static_cast<int64_t>(n * kHV + hv) * (kMaxRing * 8);
        const uint32_t cdst = smem_u32(stg + O_C);
        for (int i = lane; i < W * 2; i += 32) cp_async_16(cdst + 16 * i, csrc + 4 * i);
      }
      if (lane < kScalars / 4) {
        const float* ssrc = p.ws_sc + static_cast<int64_t>(n * kHV + hv) * kScalars;
        cp_async_16(smem_u32(stg + O_S) + 16 * lane, ssrc + 4 * lane);
      }
      cp_async_mbar_arrive_noinc(full[s]);
    };

    // Tile i of this CTA: its issue mode and metadata (fill() also publishes info[s]).
    struct TM { int g, hv, bosn, ntn, wpn, cbn, mode; int64_t sdn, rin; };
    auto fill = [&](int i, int s, TM& m) {
      const int lk = i - i0;
      m.hv = m.g % kHV;
      m.bosn = __shfl_sync(kFull, m_bos, lk);
      m.ntn = __shfl_sync(kFull, m_nt, lk);
      m.wpn = __shfl_sync(kFull, m_wp, lk);
      m.cbn = __shfl_sync(kFull, m_cb, lk);
      const int fln = __shfl_sync(kFull, m_fl, lk);
      m.sdn = __shfl_sync(kFull, m_sd, lk);
      m.rin = __shfl_sync(kFull, m_ri, lk);
      const bool live = fln == 0, has_state = m.sdn >= 0;
      m.mode = (live && m.ntn > 0) ? (has_state ? 1 : 2) : 0;
      if (lane == 0) {
        if (m.mode == 1) info[s] = Info{1, m.ntn, m.bosn, static_cast<int>(m.rin), m.wpn, m.cbn, m.wpn, Lm};
        else if (m.mode == 2) info[s] = Info{2, m.ntn, m.bosn, 0, 0, 0, 0, Lm};  // NULL state
        else info[s] = Info{0, 0, 0, 0, 0, 0, 0, Lm};                             // skip
      }
    };

    // ---- prologue: state tiles of the first NST stages, ahead of prep completion ---------------
    ld_meta(0);
    int npre = 0;
#pragma unroll
    for (int i = 0; i < NST; ++i) {
      const int g = blockIdx.x + i * G;
      if (g >= ntiles) break;
      TM m;
      m.g = g;
      fill(i, i, m);
      if (m.mode == 1) issue_state(m.hv, i, m.sdn);
      ++npre;
    }
    pdl_wait();  // prep grid done: everything below may read its workspace

    for (int i = 0;; ++i) {
      const int g = blockIdx.x + i * G;
      if (g >= ntiles) break;
      const int s = i % NST;
      TM m;
      m.g = g;
      if (i >= npre) {  // stages beyond the prologue: metadata window refill + empty wait
        if (i - i0 >= kMB || i < i0) ld_meta(i);
        mbar_wait(empty[s], ((i / NST) & 1) ^ 1);
        fill(i, s, m);
        if (m.mode == 1) issue_state(m.hv, s, m.sdn);
      } else {
        fill(i, s, m);  // same values as the prologue pass (info[s] rewritten identically)
      }
      if (m.mode == 1) {
        issue_aux(m.hv, s, g / kHV, m.rin, m.bosn, m.ntn, m.wpn, m.cbn);
      } else {
        mbar_arrive(full[s]);                // 32 lane arrives + 1 extra = 33
        if (lane == 0) mbar_arrive(full[s]);
      }
    }
    return;
  }

  // ---- consumers (warps 0..3): thread t = state row t ---------------------------------------
  for (int i = 0;; ++i) {
    const int g = blockIdx.x + i * G;
    if (g >= ntiles) break;
    const int s = i % NST;
    mbar_wait(full[s], (i / NST) & 1);
    const Info inf = info[s];
    if (inf.mode == 0) { mbar_arrive(empty[s]); continue; }
    const int hv = g % kHV;
    char* st = smem_raw + s * (SB + AUX);
    const float* vecs = reinterpret_cast<const float*>(st + SB);                    // [8][kK]
    const __nv_bfloat16* dtile = reinterpret_cast<const __nv_bfloat16*>(st + O_D);  // [32][V]
    const float* coef = reinterpret_cast<const float*>(st + O_C);                   // [32][8]
    const float* scs = reinterpret_cast<const float*>(st + O_S);                    // [kScalars]
    const __nv_bfloat16* vj = reinterpret_cast<const __nv_bfloat16*>(st + O_V);     // [4][V]

    if (inf.mode == 2) {  // NULL state block: zero the output rows, nothing else
      for (int j = 0; j < inf.nt; ++j)
        p.out[static_cast<int64_t>(inf.bos + j) * p.o_ts + hv * kV + t] =
            __float2bfloat16_rn(0.0f);
      mbar_arrive(empty[s]);
      continue;
    }

    // ---- state dots: hw[m] = S0[t] . vec_m in one pass, then the 8-lane butterfly ----------
    const ST* tile = reinterpret_cast<const ST*>(st);
    const int c8 = lane & 7;               // one of the 8 lanes sharing 8 state rows
    const ST* tl = tile + static_cast<int64_t>(32 * w + (lane >> 3) * 8) * kK + c8 * VEC;
    float acc[8][8];
    float hw[8];
#pragma unroll
    for (int j = 0; j < 8; ++j)
#pragma unroll
      for (int m = 0; m < 8; ++m) acc[j][m] = 0.0f;
#pragma unroll
    for (int i2 = 0; i2 < NI; ++i2) {
      // state columns i2*8*VEC + c8*VEC .. +VEC of rows row0..row0+7 (vecs layout as in prep)
      float4 vv[VEC / 4][8];
#pragma unroll
      for (int hh = 0; hh < VEC / 4; ++hh)
#pragma unroll
        for (int m = 0; m < 8; ++m) vv[hh][m] = ld16(vecs + m * kK + hh * 64 + (8 * i2 + c8) * 4);
      if constexpr (VEC == 4) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          float sv[4];
          Tile::unpack(Tile::load(tl + static_cast<int64_t>(j) * kK + i2 * 8 * VEC), sv);
#pragma unroll
          for (int m = 0; m < 8; ++m) {
            acc[j][m] = fmaf(sv[0], vv[0][m].x, acc[j][m]);
            acc[j][m] = fmaf(sv[1], vv[0][m].y, acc[j][m]);
            acc[j][m] = fmaf(sv[2], vv[0][m].z, acc[j][m]);
            acc[j][m] = fmaf(sv[3], vv[0][m].w, acc[j][m]);
          }
        }
      } else {  // bf16 state: one 16 B raw load covers two float4 halves of vecs
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          float sv[8];
          Tile::unpack(Tile::load(tl + static_cast<int64_t>(j) * kK + i2 * 8 * VEC), sv);
#pragma unroll
          for (int m = 0; m < 8; ++m)
#pragma unroll
            for (int hh = 0; hh < 2; ++hh) {
              acc[j][m] = fmaf(sv[4 * hh + 0], vv[hh][m].x, acc[j][m]);
              acc[j][m] = fmaf(sv[4 * hh + 1], vv[hh][m].y, acc[j][m]);
              acc[j][m] = fmaf(sv[4 * hh + 2], vv[hh][m].z, acc[j][m]);
              acc[j][m] = fmaf(sv[4 * hh + 3], vv[hh][m].w, acc[j][m]);
            }
        }
      }
    }
    // reduce over the 8 lanes of the row group; lane c8 keeps row c8's full sums
#pragma unroll
    for (int sb = 2; sb >= 0; --sb) {
      const int half = 1 << sb;
      const bool hi = (c8 >> sb) & 1;
#pragma unroll
      for (int j = 0; j < half; ++j)
#pragma unroll
        for (int m = 0; m < 8; ++m) {
          const float send = hi ? acc[j][m] : acc[j + half][m];
          const float keep = hi ? acc[j + half][m] : acc[j][m];
          acc[j][m] = keep + __shfl_xor_sync(kFull, send, half);
        }
    }
#pragma unroll
    for (int m = 0; m < 8; ++m) hw[m] = acc[0][m];

    // ---- history correction: hw *= total_decay; hw += sum_c d[c][t] * coef[c][m] ------------
    if (inf.W > 0) {
      const float total_decay = scs[28];
#pragma unroll
      for (int m = 0; m < 8; ++m) hw[m] *= total_decay;
      for (int c = 0; c < inf.W; ++c) {
        const float dd = bf2f(dtile[c * kV + t]);
        const float4 c0 = ld16(coef + c * 8), c1 = ld16(coef + c * 8 + 4);
        hw[0] = fmaf(dd, c0.x, hw[0]); hw[1] = fmaf(dd, c0.y, hw[1]);
        hw[2] = fmaf(dd, c0.z, hw[2]); hw[3] = fmaf(dd, c0.w, hw[3]);
        hw[4] = fmaf(dd, c1.x, hw[4]); hw[5] = fmaf(dd, c1.y, hw[5]);
        hw[6] = fmaf(dd, c1.z, hw[6]); hw[7] = fmaf(dd, c1.w, hw[7]);
      }
    }

    // ---- tail: D, O and the output / ring stores (scs precomputed by prep) ------------------
    float vjr[kMaxSpec];
#pragma unroll
    for (int j = 0; j < kMaxSpec; ++j) vjr[j] = bf2f(vj[j * kV + t]);
    float Rj[kMaxSpec], D[kMaxSpec];
#pragma unroll
    for (int j = 0; j < kMaxSpec; ++j) Rj[j] = scs[6 + j] * (vjr[j] - scs[10 + j] * hw[4 + j]);
#pragma unroll
    for (int i2 = 0; i2 < kMaxSpec; ++i2) {
      float s2 = Rj[i2];  // T[i][i] = 1
#pragma unroll
      for (int j = 0; j < i2; ++j) s2 = fmaf(Rj[j], scs[i2 * (i2 - 1) / 2 + j], s2);
      D[i2] = s2;
    }
    // Everything the stores need comes out of smem before the release: once all 128 consumers
    // arrive, the producer refills this stage (scs included) for tile i + NST.
    float o[kMaxSpec], gj[kMaxSpec];
#pragma unroll
    for (int j = 0; j < kMaxSpec; ++j) {
      o[j] = scs[10 + j] * hw[j];
#pragma unroll
      for (int i2 = 0; i2 <= j; ++i2) o[j] = fmaf(D[i2], scs[18 + j * (j + 1) / 2 + i2], o[j]);
      gj[j] = scs[14 + j];
    }
    mbar_arrive(empty[s]);  // stage released: every smem read of this tile is done
#pragma unroll
    for (int j = 0; j < kMaxSpec; ++j) {
      if (j >= inf.nt) break;
      p.out[static_cast<int64_t>(inf.bos + j) * p.o_ts + hv * kV + t] = __float2bfloat16_rn(o[j]);
      if (inf.wp + j < p.L) {
        const int phys = (inf.base + inf.wp + j) & inf.Lm;
        const __nv_bfloat16 hi = __float2bfloat16_rn(D[j]);
        const __nv_bfloat16 lo = __float2bfloat16_rn(D[j] - bf2f(hi));
        p.d_cache[inf.r * p.d_ss + (static_cast<int64_t>(hv) * p.L + phys) * kV + t] = hi;
        p.rawv_cache[inf.r * p.rawv_ss + (static_cast<int64_t>(hv) * p.L + phys) * kV + t] = lo;
        if (t == 0) p.g_cache[inf.r * p.g_ss + static_cast<int64_t>(hv) * p.L + phys] = gj[j];
      }
    }
  }
}

}  // namespace gdn
}  // namespace green
