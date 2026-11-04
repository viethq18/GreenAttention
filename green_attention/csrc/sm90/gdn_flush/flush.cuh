// SPDX-License-Identifier: Apache-2.0
// SM90 Replay-SSM commit ("flush"): materialize the fp32 checkpoint from the compact D/K/G ring.
//
// Replaces, per verify step, the Triton pair
//   gdn_replayssm_compact_commit_kernel  (grid (2, B, HV*L), one CTA per 64 v-columns of one head)
//   _finish_gdn_replayssm_circular_fold_kernel
// of sglang/kernels/ops/attention/linear/gdn_replayssm_spec_decode.py (contract: docs/gdn-contract.md §7).
//
// Per (fold|track job, layer, key head h) unit one CTA updates the three value heads hv = 3h..3h+2 that
// share k-head h:   H' = H * exp(total) + sum_t decay_t * (K_t + Kres_t)^T (d_t + dres_t),
//   decay_t = exp(total - prefix_t) for t < len else 0,  u = (d+dres) * decay,  hi = bf16(u),
//   lo = bf16(u - hi),  delta = Kt.hi + Kt.lo + Kresᵀ.hi + Kresᵀ.lo   (bf16 mma, fp32 accumulate).
// The state is streamed straight through registers in the accumulator fragment layout.
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace green::gdn_flush {

constexpr int kH = 16;      // key heads
constexpr int kHV = 48;     // value heads
constexpr int kGrp = 3;     // value heads per key head
constexpr int kK = 128;
constexpr int kV = 128;
constexpr int kT = 32;      // ring length (MAX_CACHE_LEN)
constexpr int kThreads = 256;
constexpr int kWarps = 8;
constexpr int kLd = kK + 8;  // smem row stride (elements): 272 B -> conflict-free ldmatrix
constexpr int kMtPerHv = kV / 16;
constexpr int kMt = kGrp * kMtPerHv;  // 24 m16 tiles per unit
constexpr int kDepth = 6;             // state m-tiles in flight per warp
constexpr int kMaxRows = 1024;

// One fold and/or track job. fold_len <= 0: no fold pass. track_idx < 0: no track pass.
struct Job {
  int state_idx;
  int replay_idx;
  int base;
  int fold_len;
  int track_idx;
  int track_len;
  int pad0;
  int pad1;
};

struct Params {
  // checkpoint [L, S, HV, V, K] fp32; strides in elements
  float* state;
  int64_t state_ls, state_ss;
  // ring tensors [L, S2, ...] bf16 / fp32
  const __nv_bfloat16* d;
  const __nv_bfloat16* dres;
  const __nv_bfloat16* k;
  const __nv_bfloat16* kres;
  const float* g;
  int64_t d_ls, d_ss, dres_ls, dres_ss, k_ls, k_ss, kres_ls, kres_ss, g_ls, g_ss;
  int layers;
  // prep inputs
  const void* state_idx;
  const void* replay_idx;
  const void* accept;
  const void* track_idx;
  const void* track_step;
  int* write_pos;
  int* cache_base;
  int8_t* is_flush;
  int64_t state_idx_stride, replay_idx_stride, accept_stride, track_idx_stride, track_step_stride;
  int state_idx_bits, replay_idx_bits, accept_bits, track_idx_bits, track_step_bits;
  int has_track;
  int rows;
  int null_id;
  // work list
  Job* jobs;
  int* counters;  // [0] = number of jobs, [1] = next unit
};

__device__ __forceinline__ int64_t load_index(const void* p, int bits, int64_t i) {
  return bits == 64 ? static_cast<const int64_t*>(p)[i] : static_cast<int64_t>(static_cast<const int32_t*>(p)[i]);
}

// ---------------------------------------------------------------------------------------------
// prep: one CTA. Builds the compact job list, snapshots the cursors into it and applies the
// post-fold cursor reset (the Triton "finish" kernel), so the fold kernel never reads cursors.
__global__ void __launch_bounds__(256) prep_kernel(Params p) {
  __shared__ int warp_sum[8];
  __shared__ int carry;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  if (tid == 0) carry = 0;
  __syncthreads();
  for (int c0 = 0; c0 < p.rows; c0 += 256) {
    const int i = c0 + tid;
    bool job = false;
    Job j{};
    if (i < p.rows) {
      const int64_t state_idx = load_index(p.state_idx, p.state_idx_bits, i * p.state_idx_stride);
      const int64_t replay = load_index(p.replay_idx, p.replay_idx_bits, i * p.replay_idx_stride);
      if (state_idx > p.null_id) {
        const int n_hist = p.write_pos[replay];
        const bool fold = p.is_flush[replay] != 0;
        const int base = p.cache_base[replay];
        int track_idx = -1, track_len = 0;
        if (p.has_track) {
          const int64_t t_idx = load_index(p.track_idx, p.track_idx_bits, i * p.track_idx_stride);
          const int64_t t_step = load_index(p.track_step, p.track_step_bits, i * p.track_step_stride);
          if (t_idx > p.null_id && t_step >= 0) {
            const int64_t acc = load_index(p.accept, p.accept_bits, i * p.accept_stride);
            track_idx = static_cast<int>(t_idx);
            track_len = static_cast<int>(n_hist - acc + t_step + 1);
          }
        }
        const int fold_len = fold ? n_hist : 0;
        j.state_idx = static_cast<int>(state_idx);
        j.replay_idx = static_cast<int>(replay);
        j.base = base;
        j.fold_len = fold_len;
        j.track_idx = track_idx;
        j.track_len = track_len;
        job = (fold_len > 0) || (track_idx >= 0);
        if (fold) {  // _finish_gdn_replayssm_circular_fold_kernel
          p.cache_base[replay] = (base + n_hist) & (kT - 1);
          p.write_pos[replay] = 0;
          p.is_flush[replay] = 0;
        }
      }
    }
    const unsigned ballot = __ballot_sync(~0u, job);
    const int rank_in_warp = __popc(ballot & ((1u << lane) - 1));
    if (lane == 0) warp_sum[warp] = __popc(ballot);
    __syncthreads();
    int before = carry;
    int total = 0;
    for (int w = 0; w < 8; ++w) {
      if (w < warp) before += warp_sum[w];
      total += warp_sum[w];
    }
    if (job) p.jobs[before + rank_in_warp] = j;
    __syncthreads();
    if (tid == 0) carry += total;
    __syncthreads();
  }
  if (tid == 0) {
    p.counters[0] = carry;
    p.counters[1] = 0;
  }
}

// ---------------------------------------------------------------------------------------------
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void cp_async16(void* dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem_u32(dst)), "l"(src));
}

__device__ __forceinline__ void ldsm_x4_trans(uint32_t (&r)[4], const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(smem_u32(p)));
}

__device__ __forceinline__ void mma_bf16(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ float2 ld_state(const float* p) {
  float2 v;
  asm volatile("ld.global.L1::no_allocate.v2.f32 {%0,%1}, [%2];\n" : "=f"(v.x), "=f"(v.y) : "l"(p));
  return v;
}

struct Smem {
  __nv_bfloat16 k[kT][kLd];
  __nv_bfloat16 kres[kT][kLd];
  __nv_bfloat16 hi[kGrp][kT][kLd];
  __nv_bfloat16 lo[kGrp][kT][kLd];
  float decay[kGrp][kT];
  float et[kGrp];
  int unit;
};

template <bool HAS_RES>
struct Pass {
  // One update pass of a unit: reads old state from `src_slot`, writes the new state to `dst_slot`.
  __device__ static void run(const Params& p, Smem& s, const Job& job, int layer, int h, int len,
                             int dst_slot) {
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g8 = lane >> 2, q = lane & 3;
  const int replay = job.replay_idx;

  // ---- gates: warps 0..2 each scan one value head --------------------------------------------
  if (warp < kGrp) {
    const int hv = h * kGrp + warp;
    const int phys = (job.base + lane) & (kT - 1);
    float gate = p.g[layer * p.g_ls + static_cast<int64_t>(replay) * p.g_ss + hv * kT + phys];
    float x = lane < len ? gate : 0.f;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const float y = __shfl_up_sync(~0u, x, o);
      if (lane >= o) x += y;
    }
    const float total = __shfl_sync(~0u, x, 31);
    s.decay[warp][lane] = lane < len ? __expf(total - x) : 0.f;
    if (lane == 0) s.et[warp] = __expf(total);
  }

  // ---- state prefetch (independent of the prep below) ----------------------------------------
  float* sbase0 = p.state + layer * p.state_ls + static_cast<int64_t>(job.state_idx) * p.state_ss +
                  static_cast<int64_t>(h * kGrp) * (kV * kK) + (warp * 16 + 2 * q);
  float2 st[kDepth][4];
#pragma unroll
  for (int m = 0; m < kDepth; ++m) {
    const float* sb = sbase0 + static_cast<int64_t>((m / kMtPerHv) * (kV * kK) + (m % kMtPerHv) * 16 * kK + g8 * kK);
    st[m][0] = ld_state(sb);
    st[m][1] = ld_state(sb + 8 * kK);
    st[m][2] = ld_state(sb + 8);
    st[m][3] = ld_state(sb + 8 * kK + 8);
  }

  // ---- D tiles -> registers (12 x 16 B per thread) ---------------------------------------------
  uint4 draw[kGrp][2], rraw[kGrp][2];
#pragma unroll
  for (int hvi = 0; hvi < kGrp; ++hvi) {
#pragma unroll
    for (int jj = 0; jj < 2; ++jj) {
      const int c = tid + jj * kThreads;
      const int t = c >> 4, v8 = (c & 15) * 8;
      const int phys = (job.base + t) & (kT - 1);
      const int64_t off = (static_cast<int64_t>(h * kGrp + hvi) * kT + phys) * kV + v8;
      draw[hvi][jj] = *reinterpret_cast<const uint4*>(p.d + layer * p.d_ls + static_cast<int64_t>(replay) * p.d_ss + off);
      if (HAS_RES)
        rraw[hvi][jj] = *reinterpret_cast<const uint4*>(p.dres + layer * p.dres_ls + static_cast<int64_t>(replay) * p.dres_ss + off);
    }
  }
  __syncthreads();  // decay / et visible (and the previous pass is done with smem)

#pragma unroll
  for (int hvi = 0; hvi < kGrp; ++hvi) {
#pragma unroll
    for (int jj = 0; jj < 2; ++jj) {
      const int c = tid + jj * kThreads;
      const int t = c >> 4, v8 = (c & 15) * 8;
      const float dec = s.decay[hvi][t];
      const __nv_bfloat162* dv = reinterpret_cast<const __nv_bfloat162*>(&draw[hvi][jj]);
      const __nv_bfloat162* rv = reinterpret_cast<const __nv_bfloat162*>(&rraw[hvi][jj]);
      uint32_t hi_w[4], lo_w[4];
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        float2 u = __bfloat1622float2(dv[e]);
        if (HAS_RES) {
          const float2 r = __bfloat1622float2(rv[e]);
          u.x += r.x;
          u.y += r.y;
        }
        u.x *= dec;
        u.y *= dec;
        const __nv_bfloat162 hi = __floats2bfloat162_rn(u.x, u.y);
        const float2 hf = __bfloat1622float2(hi);
        const __nv_bfloat162 lo = __floats2bfloat162_rn(u.x - hf.x, u.y - hf.y);
        hi_w[e] = *reinterpret_cast<const uint32_t*>(&hi);
        lo_w[e] = *reinterpret_cast<const uint32_t*>(&lo);
      }
      *reinterpret_cast<uint4*>(&s.hi[hvi][t][v8]) = make_uint4(hi_w[0], hi_w[1], hi_w[2], hi_w[3]);
      *reinterpret_cast<uint4*>(&s.lo[hvi][t][v8]) = make_uint4(lo_w[0], lo_w[1], lo_w[2], lo_w[3]);
    }
  }
  asm volatile("cp.async.wait_all;\n" ::: "memory");
  __syncthreads();

  // ---- B fragments: this warp's 16 key columns, both k16 chunks, K and K-residual --------------
  const int mi = lane >> 3, r8 = lane & 7;
  uint32_t bk[2][4], bkr[2][4];
#pragma unroll
  for (int ch = 0; ch < 2; ++ch) {
    const int t = ch * 16 + r8 + ((mi & 1) ? 8 : 0);
    const int col = warp * 16 + ((mi >> 1) ? 8 : 0);
    ldsm_x4_trans(bk[ch], &s.k[t][col]);
    if (HAS_RES) ldsm_x4_trans(bkr[ch], &s.kres[t][col]);
  }

  // ---- m-tile loop -----------------------------------------------------------------------------
  float* dbase0 = p.state + layer * p.state_ls + static_cast<int64_t>(dst_slot) * p.state_ss +
                  static_cast<int64_t>(h * kGrp) * (kV * kK) + (warp * 16 + 2 * q);
#pragma unroll
  for (int m = 0; m < kMt; ++m) {
    const int hvi = m / kMtPerHv, v0 = (m % kMtPerHv) * 16;
    float acc0[4] = {0.f, 0.f, 0.f, 0.f}, acc1[4] = {0.f, 0.f, 0.f, 0.f};
    uint32_t ahi[2][4], alo[2][4];
#pragma unroll
    for (int ch = 0; ch < 2; ++ch) {
      const int t = ch * 16 + r8 + ((mi >> 1) ? 8 : 0);
      const int v = v0 + ((mi & 1) ? 8 : 0);
      ldsm_x4_trans(ahi[ch], &s.hi[hvi][t][v]);
      ldsm_x4_trans(alo[ch], &s.lo[hvi][t][v]);
    }
    // order: K.hi, K.lo, Kres.hi, Kres.lo (each over both k16 chunks), as the Triton dot chain
#pragma unroll
    for (int ch = 0; ch < 2; ++ch) {
      mma_bf16(acc0, ahi[ch], bk[ch][0], bk[ch][1]);
      mma_bf16(acc1, ahi[ch], bk[ch][2], bk[ch][3]);
    }
#pragma unroll
    for (int ch = 0; ch < 2; ++ch) {
      mma_bf16(acc0, alo[ch], bk[ch][0], bk[ch][1]);
      mma_bf16(acc1, alo[ch], bk[ch][2], bk[ch][3]);
    }
    if (HAS_RES) {
#pragma unroll
      for (int ch = 0; ch < 2; ++ch) {
        mma_bf16(acc0, ahi[ch], bkr[ch][0], bkr[ch][1]);
        mma_bf16(acc1, ahi[ch], bkr[ch][2], bkr[ch][3]);
      }
#pragma unroll
      for (int ch = 0; ch < 2; ++ch) {
        mma_bf16(acc0, alo[ch], bkr[ch][0], bkr[ch][1]);
        mma_bf16(acc1, alo[ch], bkr[ch][2], bkr[ch][3]);
      }
    }
    const float et = s.et[hvi];
    const int slot = m % kDepth;
    float* db = dbase0 + static_cast<int64_t>(hvi * (kV * kK) + v0 * kK + g8 * kK);
    float2 o;
    o = make_float2(fmaf(st[slot][0].x, et, acc0[0]), fmaf(st[slot][0].y, et, acc0[1]));
    *reinterpret_cast<float2*>(db) = o;
    o = make_float2(fmaf(st[slot][1].x, et, acc0[2]), fmaf(st[slot][1].y, et, acc0[3]));
    *reinterpret_cast<float2*>(db + 8 * kK) = o;
    o = make_float2(fmaf(st[slot][2].x, et, acc1[0]), fmaf(st[slot][2].y, et, acc1[1]));
    *reinterpret_cast<float2*>(db + 8) = o;
    o = make_float2(fmaf(st[slot][3].x, et, acc1[2]), fmaf(st[slot][3].y, et, acc1[3]));
    *reinterpret_cast<float2*>(db + 8 * kK + 8) = o;
    // refill the slot just consumed with m-tile m + kDepth
    if (m + kDepth < kMt) {
      const int mn = m + kDepth;
      const float* sb = sbase0 + static_cast<int64_t>((mn / kMtPerHv) * (kV * kK) + (mn % kMtPerHv) * 16 * kK + g8 * kK);
      st[slot][0] = ld_state(sb);
      st[slot][1] = ld_state(sb + 8 * kK);
      st[slot][2] = ld_state(sb + 8);
      st[slot][3] = ld_state(sb + 8 * kK + 8);
    }
  }
  __syncthreads();  // smem free for the next pass / unit
  }
};

template <bool HAS_RES>
__global__ void __launch_bounds__(kThreads, 2) fold_kernel(Params p) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  Smem& s = *reinterpret_cast<Smem*>(smem_raw);
  const int tid = threadIdx.x;
  const int njobs = *reinterpret_cast<volatile const int*>(p.counters);
  const int total = njobs * p.layers * kH;
  while (true) {
    if (tid == 0) s.unit = atomicAdd(p.counters + 1, 1);
    __syncthreads();
    const int u = s.unit;
    if (u >= total) return;
    const int job_i = u / (p.layers * kH);
    const int rem = u - job_i * (p.layers * kH);
    const int layer = rem / kH, h = rem - layer * kH;
    const Job job = p.jobs[job_i];

    // K and K-residual rows (gathered in logical order t -> physical (base + t) & (T-1))
    for (int c = tid; c < kT * 16; c += kThreads) {
      const int t = c >> 4, k8 = (c & 15) * 8;
      const int phys = (job.base + t) & (kT - 1);
      const int64_t row = static_cast<int64_t>(h * kT + phys) * kK + k8;
      cp_async16(&s.k[t][k8], p.k + layer * p.k_ls + static_cast<int64_t>(job.replay_idx) * p.k_ss + row);
      if (HAS_RES)
        cp_async16(&s.kres[t][k8], p.kres + layer * p.kres_ls + static_cast<int64_t>(job.replay_idx) * p.kres_ss + row);
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");

    // track pass first: it reads the *old* checkpoint that the fold pass then overwrites
    if (job.track_idx >= 0) Pass<HAS_RES>::run(p, s, job, layer, h, job.track_len, job.track_idx);
    if (job.fold_len > 0) Pass<HAS_RES>::run(p, s, job, layer, h, job.fold_len, job.state_idx);
  }
}

}  // namespace green::gdn_flush
