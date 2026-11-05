// SM90 FP8-KV paged attention for decode / DFLASH verify (small q_len), templated on Cfg<GQA, DSPLIT>.
// Instances: Qwen3.8 (GQA 6, head_dim 256), Gemma-4 sliding (GQA 2, head_dim 256, window) and Gemma-4
// global (GQA 8, head_dim 512 = DSPLIT 2 x 256). See docs/gemma4.md for the design brief.
// Prefill / extend over an FP8 cache (Gemma-4 global) is a separate kernel sharing these helpers: extend_kernel.cuh.
//
// One CTA = (split, kv_head x D-half, sequence). The GQA q-heads x q_len tokens that share the kv head are
// packed into the M dimension of one 64-row WGMMA tile (row = token * GQA + head; rows >= GQA*q_len
// are padding). Structure follows FlashAttention-3 hopper/ (BSD-3-Clause, Copyright (c) 2022, the
// respective contributors, as shown by the AUTHORS file): warp specialization with mbarrier
// pipelines and setmaxnreg, FP8 WGMMA for both GEMMs, online softmax in fp32. Unlike FA3 the loader
// warpgroup (warps 0-3) gathers K/V tokens with 16 B cp.async (any page_size), a second warpgroup
// (warps 4-7) transposes V in registers into the K-major layout the FP8 PV WGMMA needs, and the
// consumer warpgroup (warps 8-11) does QK^T / softmax / PV.
//
// head_dim 512 (DSPLIT = 2): the two CTAs of a kv head (adjacent blockIdx.y) both compute the full
// S = Q K^T over 512 dims, and each owns 256 output dims (V gather / transpose / PV are the 256-dim path).
//
// Sliding window (window_left = W >= 0): query row at absolute position p sees keys [p - W, p]; tiles
// wholly below the first row's window are skipped (and not counted by the split planner).
//
// Smem map (bytes, 1024-aligned base), HD = 256 * DSPLIT:
//   sQ   HD/128 atoms x 64 rows x 128 B, SW128
//   sK   K_STAGES x (HD/128 atoms x 128 tokens x 128 B), SW128
//   sVr  V_STAGES x 32K   raw V tile (this CTA's 256 dims), token-major 256 B rows, chunk^f(token) swizzle
//   sVT  1 x 32K          transposed V: 256 dim-rows x 128 B (tokens), SW128, columns in
//                         P-fragment order (see transpose_v)
//   barriers, plan
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <utility>

#include <cute/arch/mma_sm90_gmma.hpp>

#include "common/ptx.cuh"

namespace green {
namespace attn_decode {

constexpr int TILE_N = 128;
constexpr int M_TILE = 64;
constexpr float REF_WINDOW = 20.f;  // log2 units: max distance between the true running max and the P reference
constexpr int VT_STAGES = 1;
constexpr int THREADS = 384;
constexpr int WS_ROWS = 64;   // rows per CTA in the split workspace
constexpr int WS_HD = 256;    // output dims per CTA (and per workspace row)
constexpr uint32_t SMEM_LIMIT = 232448;  // 227 KiB opt-in dynamic smem on SM90
constexpr int NO_LOWER = -(1 << 30);     // "no window lower bound" sentinel

// WINDOW: the instance supports a sliding window (window_left >= 0). Compile-time, because even a never-taken
// runtime lower-bound mask costs ~5-7 % of full-attention decode time (the 64-element mask/max loop codegen
// changes); full-attention instances compile exactly the causal-only mask.
template <int GQA_, int DSPLIT_, bool WINDOW_>
struct Cfg {
  static constexpr int GQA = GQA_;
  static constexpr int DSPLIT = DSPLIT_;
  static constexpr bool WINDOW = WINDOW_;
  static constexpr int HD = WS_HD * DSPLIT;  // q / k head dim
  static constexpr int KCH = HD / 16;        // 16 B chunks per K row
  static constexpr int MAX_Q = M_TILE / GQA;  // GQA * MAX_Q <= M_TILE
  static constexpr int K_STAGES = DSPLIT == 1 ? 3 : 2;
  static constexpr int V_STAGES = DSPLIT == 1 ? 2 : 1;
  static constexpr int VT_STAGES = 1;  // transposed V single-buffered
  static constexpr uint32_t Q_BYTES = M_TILE * HD;
  static constexpr uint32_t K_BYTES = TILE_N * HD;
  static constexpr uint32_t V_BYTES = TILE_N * WS_HD;
  static constexpr uint32_t SQ_OFF = 0;
  static constexpr uint32_t SK_OFF = SQ_OFF + Q_BYTES;
  static constexpr uint32_t SV_OFF = SK_OFF + K_STAGES * K_BYTES;
  static constexpr uint32_t SVT_OFF = SV_OFF + V_STAGES * V_BYTES;
  static constexpr uint32_t BAR_OFF = SVT_OFF + VT_STAGES * V_BYTES;
  static constexpr uint32_t PLAN_OFF = BAR_OFF + 128;  // int: tiles-per-CTA quota chosen by warp 0
  static constexpr uint32_t SMEM_BYTES = PLAN_OFF + 16;
  static constexpr int PLAN_SEQS = K_BYTES / 4;  // plan_quota tile cache (in sK, idle until the quota is set)
  static constexpr uint32_t SMEM_ALLOC = SMEM_BYTES + 1024;  // slack for 1024 B alignment
  static_assert(GQA * MAX_Q <= M_TILE && MAX_Q >= 1, "q rows");
  static_assert(DSPLIT == 1 || DSPLIT == 2, "head_dim 256 or 512");
  static_assert(SMEM_ALLOC <= SMEM_LIMIT, "smem");
  static_assert(2 * K_STAGES + 2 * V_STAGES + 2 * VT_STAGES <= 16, "barrier area");
};

using CfgQwen = Cfg<6, 1, false>;         // Qwen3.8-27B full attention: 24 q / 4 kv heads, head_dim 256
using CfgGemmaSliding = Cfg<2, 1, true>;  // Gemma-4-31B sliding: 32 q / 16 kv heads, head_dim 256
using CfgGemmaGlobal = Cfg<8, 2, false>;  // Gemma-4-31B global: 32 q / 4 kv heads, head_dim 512

struct Params {
  const void* q;  // e4m3 or bf16 [total_q, kv_heads * GQA, HD]
  int64_t q_s0, q_s1;
  const uint8_t* k;
  const uint8_t* v;
  int kv_heads;  // runtime grid dimension (grid.y = kv_heads * DSPLIT)
  int64_t k_s0, k_s1, k_s2;  // page, token-in-page, kv head strides (bytes)
  int64_t v_s0, v_s1, v_s2;
  int page_shift, page_mask;
  const int* page_table;
  int64_t pt_s0, pt_s1;
  const int* cache_seqlens;
  const int* cu_seqlens_q;
  const float* kdesc;  // [batch, kv_heads] (strided) or null
  const float* vdesc;
  int64_t kd_s0, kd_s1, vd_s0, vd_s1;
  __nv_bfloat16* out;  // [total_q, kv_heads * GQA, HD], last-dim stride 1
  int64_t o_s0, o_s1;
  float* lse;  // [kv_heads * GQA, total_q] natural log, or null
  int64_t total_q;
  float* ws_o;    // [cta, WS_ROWS, WS_HD] fp32
  float* ws_lse;  // [cta, WS_ROWS] log2 domain
  float scale_log2;  // softmax_scale * log2(e)
  int window_left;  // sliding window (keys [p - W, p]); -1 = none
  int num_splits;  // grid.x: upper bound on splits per (sequence, kv head); the real count is planned on device
  int batch;
  int sm_count;
  int* eff;  // [batch] splits actually used per sequence (written by the main kernel, read by the combine kernel)
};

// ------------------------------------------------------------------ wgmma wrappers
template <size_t... I>
__device__ __forceinline__ void mma_qk_impl(uint64_t da, uint64_t db, float (&d)[64],
                                            cute::SM90::GMMA::ScaleOut so, std::index_sequence<I...>) {
  cute::SM90::GMMA::MMA_64x128x32_F32E4M3E4M3_SS_TN<>::fma(da, db, d[I]..., so);
}
__device__ __forceinline__ void mma_qk(uint64_t da, uint64_t db, float (&d)[64], bool accumulate) {
  mma_qk_impl(da, db, d, accumulate ? cute::SM90::GMMA::ScaleOut::One : cute::SM90::GMMA::ScaleOut::Zero,
              std::make_index_sequence<64>{});
}

template <size_t... I>
__device__ __forceinline__ void mma_pv_impl(const uint32_t (&a)[4], uint64_t db, float (&d)[128],
                                            std::index_sequence<I...>) {
  cute::SM90::GMMA::MMA_64x256x32_F32E4M3E4M3_RS_TN<>::fma(a[0], a[1], a[2], a[3], db, d[I]...,
                                                           cute::SM90::GMMA::ScaleOut::One);
}
__device__ __forceinline__ void mma_pv(const uint32_t (&a)[4], uint64_t db, float (&d)[128]) {
  mma_pv_impl(a, db, d, std::make_index_sequence<128>{});
}

// ------------------------------------------------------------------ producer
// KV tiles a sequence needs: [x, x + y). With a window, tiles wholly below the first query row's
// window (key < seqlen - qlen - W) are skipped.
template <class C>
__device__ __forceinline__ int2 seq_tiles(const Params& p, int b) {
  const int seqlen = p.cache_seqlens[b];
  const int t_end = (seqlen + TILE_N - 1) / TILE_N;
  int t_lo = 0;
  if (C::WINDOW && p.window_left >= 0) {
    const int q0 = p.cu_seqlens_q[b];
    const int qlen = min(p.cu_seqlens_q[b + 1] - q0, C::MAX_Q);
    t_lo = min(max(seqlen - qlen - p.window_left, 0) / TILE_N, t_end);
  }
  return make_int2(t_lo, t_end - t_lo);
}

// Gather one 128-token K tile (all HD dims): KCH 16 B chunks per token, 128 / KCH tokens per pass.
// For HD 256 (KCH 16) chunk / trow / tok / n are the exact expressions of load_v_tile, so the compiler
// shares the page-table address math (and the 16 clamped token indices) between the K and V gathers of a
// tile; signed `% KCH` / `/ KCH` defeated that and cost ~4 % of the Qwen kernel (loader issue slots).
template <class C>
__device__ __forceinline__ void load_k_tile(const Params& p, const int* pt, int kvh, int seqlen, int tile,
                                            uint32_t stage, int tid) {
  static_assert(C::KCH == 16 || C::KCH == 32, "KCH");
  constexpr int RPP = 128 / C::KCH;
  constexpr int KSH = C::KCH == 16 ? 4 : 5;
  const int chunk = tid & (C::KCH - 1);
  const int trow = tid >> KSH;  // tid < 128: 0..RPP-1
  const uint8_t* head_base = p.k + static_cast<int64_t>(kvh) * p.k_s2 + chunk * 16;
#pragma unroll
  for (int i = 0; i < TILE_N / RPP; ++i) {
    const int tok = trow + RPP * i;
    const int n = min(tile * TILE_N + tok, seqlen - 1);  // clamp: masked tail rows read real data
    const int64_t page = pt[static_cast<int64_t>(n >> p.page_shift) * p.pt_s1];
    const uint8_t* src = head_base + page * p.k_s0 + static_cast<int64_t>(n & p.page_mask) * p.k_s1;
    const int sw = RPP == 8 ? trow : (tok & 7);  // SW128 row phase = tok & 7 (= trow when RPP == 8)
    const uint32_t dst = stage + (chunk >> 3) * 16384 + tok * 128 + (((chunk & 7) ^ sw) << 4);
    cp_async_16(dst, src);
  }
}

// Gather one 128-token V tile, dims [dh*256, dh*256+256) (this thread: 16 tokens x one 16 B chunk).
__device__ __forceinline__ void load_v_tile(const Params& p, const int* pt, int kvh, int dh, int seqlen, int tile,
                                            uint32_t stage, int tid) {
  const int chunk = tid & 15;
  const int trow = tid >> 4;  // 0..7
  const uint8_t* head_base = p.v + static_cast<int64_t>(kvh) * p.v_s2 + dh * WS_HD + chunk * 16;
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    const int tok = trow + 8 * i;
    const int n = min(tile * TILE_N + tok, seqlen - 1);
    const int64_t page = pt[static_cast<int64_t>(n >> p.page_shift) * p.pt_s1];
    const uint8_t* src = head_base + page * p.v_s0 + static_cast<int64_t>(n & p.page_mask) * p.v_s1;
    const int f = ((trow >> 1) & 3) | (((i >> 1) & 1) << 2);
    cp_async_16(stage + tok * 256 + ((chunk ^ f) << 4), src);
  }
}

// Cost-model constant for split planning: fixed per-CTA overhead (prologue, pipeline fill, epilogue),
// expressed in KV tiles of work.
constexpr float PLAN_C0 = 2.0f;

// Choose the number of KV tiles each CTA processes (the "quota") from the real cache_seqlens (and the
// window), so that the grid of units * sum_b ceil(tiles_b / quota) CTAs (units = kv_heads * DSPLIT = grid.y)
// fills the GPU without wave-quantization losses. Candidate s = 1..num_splits gives quota ceil(max_tiles / s);
// the estimated makespan is min(waves * (quota + c0), total / SMs + quota + c0) (uniform-wave bound vs.
// list-scheduling bound). Ties go to fewer splits. Must be called by one full warp; all lanes get the result.
// Depends only on device memory, so it is CUDA-graph safe. `tiles` caches the first C::PLAN_SEQS tile counts:
// one lane per candidate re-reading cache_seqlens serially cost every CTA 8-33 us at batch 16-128 (H200).
template <class C>
__device__ __forceinline__ int plan_quota(const Params& p, int lane, int* tiles) {
  int mx = 0, tot = 0;
  for (int b = lane; b < p.batch; b += 32) {
    const int t = seq_tiles<C>(p, b).y;
    if (b < C::PLAN_SEQS) tiles[b] = t;
    mx = max(mx, t);
    tot += t;
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    mx = max(mx, __shfl_xor_sync(0xffffffffu, mx, o));
    tot += __shfl_xor_sync(0xffffffffu, tot, o);
  }
  if (mx <= 1) return 1;
  __syncwarp();
  const int units = static_cast<int>(gridDim.y);
  const float total = static_cast<float>(units * tot);
  const float sms = static_cast<float>(p.sm_count);
  const int ns = p.num_splits;
  // `width` candidates per pass; the 32 / width lanes sharing a candidate split its batch loop.
  const int width = ns >= 32 ? 32 : 1 << (32 - __clz(ns - 1));
  const int b_step = 32 / width;
  float best = INFINITY;
  int best_s = 1;
  for (int s0 = 1; s0 <= ns; s0 += width) {
    const int s = s0 + lane % width;
    const int quota = (mx + s - 1) / s;
    int nc = 0;
#pragma unroll 4
    for (int b = lane / width; b < p.batch; b += b_step) {
      const int t = b < C::PLAN_SEQS ? tiles[b] : seq_tiles<C>(p, b).y;
      nc += max((t + quota - 1) / quota, 1);
    }
    for (int o = 16; o >= width; o >>= 1) nc += __shfl_xor_sync(0xffffffffu, nc, o);
    if (s > ns) continue;
    nc *= units;
    const float job = static_cast<float>(quota) + PLAN_C0;
    const float est = fminf(ceilf(static_cast<float>(nc) / sms) * job,
                            (total + PLAN_C0 * static_cast<float>(nc)) / sms + job);
    if (est < best) {
      best = est;
      best_s = s;
    }
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    const float oe = __shfl_xor_sync(0xffffffffu, best, o);
    const int os = __shfl_xor_sync(0xffffffffu, best_s, o);
    if (oe < best || (oe == best && os < best_s)) {
      best = oe;
      best_s = os;
    }
  }
  return (mx + best_s - 1) / best_s;
}

// Transpose the raw V tile [128 tokens][256 dims] into sVT [256 dims][128 tokens], with the token
// columns permuted so that the P accumulator fragment of the QK^T wgmma is directly the A operand
// of the PV wgmma. Logical k index L of k-step kk (tokens 32kk..32kk+31 of the *sVT row*) holds
// physical token 32kk + 16*(L/16) + {2q, 2q+1, 8+2q, 9+2q}[L%4], q = (L%16)/4.
// Lane = q + 4*kidx handles the 4 tokens of 16-token group kidx; warp w handles dim chunks 4w..4w+3.
__device__ __forceinline__ void transpose_v(uint32_t vraw, uint32_t vt, int warp_p, int lane) {
  const int q = lane & 3;
  const int kidx = lane >> 2;
  const int t0 = 16 * kidx + 2 * q;
  const int f = ((t0 >> 1) & 3) | (((t0 >> 4) & 1) << 2);
#pragma unroll
  for (int ci = 0; ci < 4; ++ci) {
    const int c = 4 * warp_p + ci;
    const uint32_t col = static_cast<uint32_t>((c ^ f) << 4);
    uint32_t w[4][4];
    const int toks[4] = {t0, t0 + 1, t0 + 8, t0 + 9};
#pragma unroll
    for (int m = 0; m < 4; ++m) {
      const uint32_t addr = vraw + toks[m] * 256 + col;
      asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n"
                   : "=r"(w[m][0]), "=r"(w[m][1]), "=r"(w[m][2]), "=r"(w[m][3])
                   : "r"(addr));
    }
#pragma unroll
    for (int jj = 0; jj < 4; ++jj) {
      const uint32_t t0r = prmt(w[0][jj], w[1][jj], 0x5140);
      const uint32_t t1r = prmt(w[0][jj], w[1][jj], 0x7362);
      const uint32_t t2r = prmt(w[2][jj], w[3][jj], 0x5140);
      const uint32_t t3r = prmt(w[2][jj], w[3][jj], 0x7362);
      const uint32_t o[4] = {prmt(t0r, t2r, 0x5410), prmt(t0r, t2r, 0x7632), prmt(t1r, t3r, 0x5410),
                             prmt(t1r, t3r, 0x7632)};
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int d = 16 * c + 4 * jj + i;
        const uint32_t addr = vt + d * 128 + (((kidx ^ (d & 7))) << 4) + 4 * q;
        asm volatile("st.shared.b32 [%0], %1;\n" ::"r"(addr), "r"(o[i]) : "memory");
      }
    }
  }
}
// ------------------------------------------------------------------ softmax half-tile
// P rounding note: P is quantized to e4m3 (x256) relative to a reference exponent `m_run` that is chosen per
// 64-column half-tile as max(half-tile row max, running max - REF_WINDOW). The half-tile's own largest element is
// then exactly 1.0 (exact in e4m3), which matches FA4-SM90's tile_n = 64 + split-KV numerics on peaked rows,
// whereas a reference chained across 128-column tiles loses those exact peaks. `m_top` is the true running max
// (bounds how far the reference may sit below it, so P never underflows relative to the dominant keys);
// acc_o / l_run are always expressed in the reference `m_run`, so lse2 = m_run + log2(l_run) in the epilogue.
// preg gets the e4m3x4 registers of columns [64H, 64H+64); al0/al1 are the acc_o rescale factors for rows 0/1.
template <int H>
__device__ __forceinline__ void softmax_half(float* acc_s, float mx0, float mx1, float (&m_top)[2],
                                             float (&m_run)[2], float (&l_run)[2], float& al0, float& al1,
                                             uint32_t (&preg)[16]) {
  const float mn0 = fmaxf(m_top[0], mx0);
  const float mn1 = fmaxf(m_top[1], mx1);
  m_top[0] = mn0;
  m_top[1] = mn1;
  const float mr0 = (mx0 == -INFINITY) ? m_run[0] : fmaxf(mx0, mn0 - REF_WINDOW);
  const float mr1 = (mx1 == -INFINITY) ? m_run[1] : fmaxf(mx1, mn1 - REF_WINDOW);
  const float mu0 = (mr0 == -INFINITY) ? 0.f : mr0;
  const float mu1 = (mr1 == -INFINITY) ? 0.f : mr1;
  al0 = ex2(m_run[0] - mu0);
  al1 = ex2(m_run[1] - mu1);
  m_run[0] = mr0;
  m_run[1] = mr1;
  float s0 = 0.f, s1 = 0.f;
#pragma unroll
  for (int i = 32 * H; i < 32 * H + 32; ++i) {
    const float pv = ex2(acc_s[i] - (((i >> 1) & 1) ? mu1 : mu0));
    acc_s[i] = pv;
    if ((i >> 1) & 1) s1 += pv; else s0 += pv;
  }
  l_run[0] = l_run[0] * al0 + s0;
  l_run[1] = l_run[1] * al1 + s1;
#pragma unroll
  for (int kk = 2 * H; kk < 2 * H + 2; ++kk) {
    const float* s = acc_s + 16 * kk;
    preg[4 * kk + 0] = cvt_e4m3x4(s[0] * 256.f, s[1] * 256.f, s[4] * 256.f, s[5] * 256.f);
    preg[4 * kk + 1] = cvt_e4m3x4(s[2] * 256.f, s[3] * 256.f, s[6] * 256.f, s[7] * 256.f);
    preg[4 * kk + 2] = cvt_e4m3x4(s[8] * 256.f, s[9] * 256.f, s[12] * 256.f, s[13] * 256.f);
    preg[4 * kk + 3] = cvt_e4m3x4(s[10] * 256.f, s[11] * 256.f, s[14] * 256.f, s[15] * 256.f);
  }
}

// ------------------------------------------------------------------ kernel
template <class C, bool Q_BF16, bool SPLIT>
__global__ void __launch_bounds__(THREADS, 1) attn_decode_kernel(const __grid_constant__ Params p) {
  extern __shared__ uint8_t smem_raw[];
  const uint32_t smem_base = (smem_u32(smem_raw) + 1023u) & ~1023u;
  uint8_t* smem_gen = smem_raw + (smem_base - smem_u32(smem_raw));

  const int split = blockIdx.x;
  const int y = blockIdx.y;
  const int kvh = y / C::DSPLIT;   // kv head
  const int dh = y % C::DSPLIT;    // which 256-dim half of V this CTA computes
  const int tid = threadIdx.x;
  if (SPLIT) griddep_launch_dependents();  // the combine grid may start launching; it waits for our completion
  const int warp = tid >> 5;
  const int lane = tid & 31;

  const int b = blockIdx.z;
  const int seqlen = p.cache_seqlens[b];
  const int q0 = p.cu_seqlens_q[b];
  const int qlen = min(p.cu_seqlens_q[b + 1] - q0, C::MAX_Q);
  // Tiles wholly below every row's window are skipped. Must match plan_quota's view (seq_tiles) or eff can
  // exceed num_splits and tail tiles are dropped.
  const int2 st = seq_tiles<C>(p, b);
  const int t_lo = st.x;
  const int ntiles = st.y;

  const uint32_t sQ = smem_base + C::SQ_OFF;
  const uint32_t sK = smem_base + C::SK_OFF;
  const uint32_t sV = smem_base + C::SV_OFF;
  const uint32_t sVT = smem_base + C::SVT_OFF;
  const uint32_t bar_k_full = smem_base + C::BAR_OFF;
  const uint32_t bar_k_empty = bar_k_full + 8 * C::K_STAGES;
  const uint32_t bar_v_full = bar_k_empty + 8 * C::K_STAGES;
  const uint32_t bar_v_empty = bar_v_full + 8 * C::V_STAGES;
  const uint32_t bar_vt_full = bar_v_empty + 8 * C::V_STAGES;
  const uint32_t bar_vt_empty = bar_vt_full + 8 * C::VT_STAGES;

  if (tid == 0) {
    for (int s = 0; s < C::K_STAGES; ++s) {
      mbar_init(bar_k_full + 8 * s, 128);
      mbar_init(bar_k_empty + 8 * s, 4);
    }
    for (int s = 0; s < C::V_STAGES; ++s) {
      mbar_init(bar_v_full + 8 * s, 128);
      mbar_init(bar_v_empty + 8 * s, 128);  // every transpose lane releases its own reads of the raw V stage
    }
    for (int s = 0; s < C::VT_STAGES; ++s) {
      mbar_init(bar_vt_full + 8 * s, 128);
      mbar_init(bar_vt_empty + 8 * s, 4);
    }
  }
  if (SPLIT && warp == 0) {
    const int q = plan_quota<C>(p, lane, reinterpret_cast<int*>(smem_gen + C::SK_OFF));
    if (lane == 0) *reinterpret_cast<int*>(smem_gen + C::PLAN_OFF) = q;
  }
  __syncthreads();
  const int quota = SPLIT ? *reinterpret_cast<const volatile int*>(smem_gen + C::PLAN_OFF) : max(ntiles, 1);
  const int eff = max((ntiles + quota - 1) / quota, 1);
  if (split >= eff) return;  // CTA-uniform; nothing was written for this slot
  if (SPLIT && split == 0 && dh == 0 && tid == 0) p.eff[b] = eff;
  const bool direct = !SPLIT || eff == 1;  // single split: write bf16 output directly
  const int t_begin = t_lo + split * quota;
  const int nt = max(min(ntiles, (split + 1) * quota) - split * quota, 0);

  if (warp < 4) {
    // ============================================================ loader warpgroup
    reg_dealloc<96>();
    if (nt > 0) {
#pragma unroll 1
      for (int lt = 0; lt < nt; ++lt) {
        const int ks = lt % C::K_STAGES;
        if (lt >= C::K_STAGES) mbar_wait(bar_k_empty + 8 * ks, ((lt / C::K_STAGES) - 1) & 1);
        load_k_tile<C>(p, p.page_table + static_cast<int64_t>(b) * p.pt_s0, kvh, seqlen, t_begin + lt,
                       sK + ks * C::K_BYTES, tid);
        cp_async_mbar_arrive_noinc(bar_k_full + 8 * ks);
        const int vs = lt % C::V_STAGES;
        if (lt >= C::V_STAGES) mbar_wait(bar_v_empty + 8 * vs, ((lt / C::V_STAGES) - 1) & 1);
        load_v_tile(p, p.page_table + static_cast<int64_t>(b) * p.pt_s0, kvh, dh, seqlen, t_begin + lt,
                    sV + vs * C::V_BYTES, tid);
        cp_async_mbar_arrive_noinc(bar_v_full + 8 * vs);
      }
    }
    return;
  }

  if (warp < 8) {
    // ============================================================ V-transpose warpgroup
    reg_dealloc<96>();
    const int warp_p = warp - 4;
#pragma unroll 1
    for (int lt = 0; lt < nt; ++lt) {
      const int vs = lt % C::V_STAGES;
      mbar_wait(bar_v_full + 8 * vs, (lt / C::V_STAGES) & 1);
      const int vts = lt % VT_STAGES;
      if (lt >= VT_STAGES) mbar_wait(bar_vt_empty + 8 * vts, ((lt / VT_STAGES) - 1) & 1);
      transpose_v(sV + vs * C::V_BYTES, sVT + vts * C::V_BYTES, warp_p, lane);
      fence_proxy_async();
      mbar_arrive(bar_vt_full + 8 * vts);
      mbar_arrive(bar_v_empty + 8 * vs);
    }
    return;
  }

  // ============================================================== consumer warpgroup
  reg_alloc<240>();
  const int ctid = tid - 256;
  const int cw = warp - 8;  // 0..3
  const int q4 = lane & 3;
  const int g = lane >> 2;
  const int valid_rows = C::GQA * qlen;
  const float kd = p.kdesc ? p.kdesc[static_cast<int64_t>(b) * p.kd_s0 + static_cast<int64_t>(kvh) * p.kd_s1] : 1.f;
  const float vd = p.vdesc ? p.vdesc[static_cast<int64_t>(b) * p.vd_s0 + static_cast<int64_t>(kvh) * p.vd_s1] : 1.f;
  const float scale2 = p.scale_log2 * kd;

  float acc_o[128];
#pragma unroll
  for (int i = 0; i < 128; ++i) acc_o[i] = 0.f;
  // m_top: true running max of the (log2-scaled) scores; m_run: the reference exponent that acc_o / l_run are
  // expressed in. m_run follows the *current tile's* max (not the running max), so every tile's largest
  // probability is exactly 1.0 and its e4m3 x256 quantization is relative to that tile (see the P rounding note
  // below). It never trails m_top by more than REF_WINDOW so acc_o stays within 2^REF_WINDOW of unit scale.
  float m_top[2] = {-INFINITY, -INFINITY};
  float m_run[2] = {-INFINITY, -INFINITY};
  float l_run[2] = {0.f, 0.f};

  const int row0 = 16 * cw + g;
  const int row1 = row0 + 8;
  // Causal upper bound (first key this row attends). Window lower bound: key >= p - W. Padding rows
  // keep the sentinel lower bound; they are never read (softmax is skipped for them) but must not
  // raise the tile's max.
  const int base = seqlen - qlen;
  const int lim0 = row0 < valid_rows ? base + row0 / C::GQA + 1 : seqlen;
  const int lim1 = row1 < valid_rows ? base + row1 / C::GQA + 1 : seqlen;
  const bool win = C::WINDOW && p.window_left >= 0;
  const int lo0 = row0 < valid_rows && win ? max(base + row0 / C::GQA - p.window_left, 0) : NO_LOWER;
  const int lo1 = row1 < valid_rows && win ? max(base + row1 / C::GQA - p.window_left, 0) : NO_LOWER;
  const int lo_max = win ? base + qlen - 1 - p.window_left : NO_LOWER;
  const bool warp_active = 16 * cw < valid_rows;

  if (nt > 0) {
    // ---- Q tile -> smem (e4m3, SW128 atoms); padding rows are zero
#pragma unroll
    for (int i = 0; i < C::KCH / 2; ++i) {
      const int u = ctid + 128 * i;
      const int r = u / C::KCH;
      const int ch = u % C::KCH;
      uint32_t w0 = 0, w1 = 0, w2 = 0, w3 = 0;
      if (r < valid_rows) {
        const int t = r / C::GQA;
        const int h = r - t * C::GQA;
        const int64_t off = static_cast<int64_t>(q0 + t) * p.q_s0 + static_cast<int64_t>(kvh * C::GQA + h) * p.q_s1;
        if (Q_BF16) {
          const uint4* src = reinterpret_cast<const uint4*>(static_cast<const __nv_bfloat16*>(p.q) + off + ch * 16);
          const uint4 a = src[0];
          const uint4 c = src[1];
          w0 = cvt_e4m3x4(bf16_lo_to_f32(a.x), bf16_hi_to_f32(a.x), bf16_lo_to_f32(a.y), bf16_hi_to_f32(a.y));
          w1 = cvt_e4m3x4(bf16_lo_to_f32(a.z), bf16_hi_to_f32(a.z), bf16_lo_to_f32(a.w), bf16_hi_to_f32(a.w));
          w2 = cvt_e4m3x4(bf16_lo_to_f32(c.x), bf16_hi_to_f32(c.x), bf16_lo_to_f32(c.y), bf16_hi_to_f32(c.y));
          w3 = cvt_e4m3x4(bf16_lo_to_f32(c.z), bf16_hi_to_f32(c.z), bf16_lo_to_f32(c.w), bf16_hi_to_f32(c.w));
        } else {
          const uint4 a = *reinterpret_cast<const uint4*>(static_cast<const uint8_t*>(p.q) + off + ch * 16);
          w0 = a.x; w1 = a.y; w2 = a.z; w3 = a.w;
        }
      }
      const uint32_t dst = sQ + (ch >> 3) * 8192 + r * 128 + (((ch & 7) ^ (r & 7)) << 4);
      asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};\n" ::"r"(dst), "r"(w0), "r"(w1), "r"(w2), "r"(w3)
                   : "memory");
    }
    fence_proxy_async();
    named_bar_sync(2, 128);

    float acc_s[64];
#pragma unroll 1
    for (int lt = 0; lt < nt; ++lt) {
      const int ks = lt % C::K_STAGES;
      const int n0 = (t_begin + lt) * TILE_N;
      mbar_wait(bar_k_full + 8 * ks, (lt / C::K_STAGES) & 1);
      fence_proxy_async();

      // ---- S = Q K^T (FP8 WGMMA, SS)
      const uint64_t dq = make_desc_sw128(sQ);
      const uint64_t dk = make_desc_sw128(sK + ks * C::K_BYTES);
      wgmma_fence();
#pragma unroll
      for (int kk = 0; kk < C::HD / 32; ++kk) {
        const uint64_t da = dq + (((kk >> 2) * 8192 + (kk & 3) * 32) >> 4);
        const uint64_t db = dk + (((kk >> 2) * 16384 + (kk & 3) * 32) >> 4);
        mma_qk(da, db, acc_s, kk > 0);
      }
      wgmma_commit();
      wgmma_wait<0>();
#pragma unroll
      for (int i = 0; i < 64; ++i) fence_operand(acc_s[i]);
      __syncwarp();
      if (lane == 0) mbar_arrive(bar_k_empty + 8 * ks);

      // ---- online softmax, P -> e4m3 (x256), in two 64-column halves. Each half quantizes P relative to its own
      // row max (see softmax_half), so the PV GEMM is issued per half with an acc_o rescale in between; half 1's
      // softmax overlaps half 0's PV GEMM.
      uint32_t preg[16];
      float al1_0 = 1.f, al1_1 = 1.f;
      float mxh[2][2];
      if (warp_active) {
        // CTA-uniform: mask when the tile reaches past the first row's causal bound or starts below the last
        // valid row's window lower bound (the largest lower bound of any row).
        const bool need_mask = (n0 + TILE_N > base + 1) || (C::WINDOW && n0 < lo_max);
        mxh[0][0] = mxh[0][1] = mxh[1][0] = mxh[1][1] = -INFINITY;
#pragma unroll
        for (int i = 0; i < 64; ++i) {
          float x = acc_s[i] * scale2;
          if (need_mask) {
            const int col = n0 + 8 * (i >> 2) + 2 * q4 + (i & 1);
            const int lim = ((i >> 1) & 1) ? lim1 : lim0;
            if (col >= lim || (C::WINDOW && col < (((i >> 1) & 1) ? lo1 : lo0))) x = -INFINITY;
          }
          acc_s[i] = x;
          float& mx = mxh[i >> 5][(i >> 1) & 1];
          mx = fmaxf(mx, x);
        }
#pragma unroll
        for (int h = 0; h < 2; ++h) {
#pragma unroll
          for (int r = 0; r < 2; ++r) {
            mxh[h][r] = fmaxf(mxh[h][r], __shfl_xor_sync(0xffffffffu, mxh[h][r], 1));
            mxh[h][r] = fmaxf(mxh[h][r], __shfl_xor_sync(0xffffffffu, mxh[h][r], 2));
          }
        }
        float al0_0, al0_1;
        softmax_half<0>(acc_s, mxh[0][0], mxh[0][1], m_top, m_run, l_run, al0_0, al0_1, preg);
#pragma unroll
        for (int i = 0; i < 128; ++i) acc_o[i] *= (((i >> 1) & 1) ? al0_1 : al0_0);
      } else {
#pragma unroll
        for (int i = 0; i < 16; ++i) preg[i] = 0u;
      }

      // ---- O += P V (FP8 WGMMA, RS)
      mbar_wait(bar_vt_full + 8 * (lt % VT_STAGES), (lt / VT_STAGES) & 1);
      const uint64_t dv = make_desc_sw128(sVT + (lt % VT_STAGES) * C::V_BYTES);
#pragma unroll
      for (int i = 0; i < 128; ++i) fence_operand(acc_o[i]);
#pragma unroll
      for (int i = 0; i < 8; ++i) fence_operand(preg[i]);
      wgmma_fence();
#pragma unroll
      for (int kk = 0; kk < 2; ++kk) {
        const uint32_t a[4] = {preg[4 * kk], preg[4 * kk + 1], preg[4 * kk + 2], preg[4 * kk + 3]};
        mma_pv(a, dv + ((kk * 32) >> 4), acc_o);
      }
      wgmma_commit();
      if (warp_active) softmax_half<1>(acc_s, mxh[1][0], mxh[1][1], m_top, m_run, l_run, al1_0, al1_1, preg);
      wgmma_wait<0>();
#pragma unroll
      for (int i = 0; i < 128; ++i) fence_operand(acc_o[i]);
      if (warp_active) {
#pragma unroll
        for (int i = 0; i < 128; ++i) acc_o[i] *= (((i >> 1) & 1) ? al1_1 : al1_0);
      }
#pragma unroll
      for (int i = 0; i < 128; ++i) fence_operand(acc_o[i]);
#pragma unroll
      for (int i = 8; i < 16; ++i) fence_operand(preg[i]);
      wgmma_fence();
#pragma unroll
      for (int kk = 2; kk < 4; ++kk) {
        const uint32_t a[4] = {preg[4 * kk], preg[4 * kk + 1], preg[4 * kk + 2], preg[4 * kk + 3]};
        mma_pv(a, dv + ((kk * 32) >> 4), acc_o);
      }
      wgmma_commit();
      wgmma_wait<0>();
#pragma unroll
      for (int i = 0; i < 128; ++i) fence_operand(acc_o[i]);
      __syncwarp();
      if (lane == 0) mbar_arrive(bar_vt_empty + 8 * (lt % VT_STAGES));
    }
  }

  // ---- epilogue
  float l0 = l_run[0], l1 = l_run[1];
  l0 += __shfl_xor_sync(0xffffffffu, l0, 1);
  l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
  l1 += __shfl_xor_sync(0xffffffffu, l1, 1);
  l1 += __shfl_xor_sync(0xffffffffu, l1, 2);
  const float os0 = l0 > 0.f ? vd * (1.f / 256.f) / l0 : 0.f;
  const float os1 = l1 > 0.f ? vd * (1.f / 256.f) / l1 : 0.f;
  const float lse2_0 = l0 > 0.f ? m_run[0] + __log2f(l0) : -INFINITY;
  const float lse2_1 = l1 > 0.f ? m_run[1] + __log2f(l1) : -INFINITY;
  const int cta = (b * static_cast<int>(gridDim.y) + y) * p.num_splits + split;

#pragma unroll
  for (int rh = 0; rh < 2; ++rh) {
    const int r = rh ? row1 : row0;
    if (r >= valid_rows) continue;
    const int t = r / C::GQA;
    const int h = r - t * C::GQA;
    const float os = rh ? os1 : os0;
    const float lse2 = rh ? lse2_1 : lse2_0;
    if (SPLIT && !direct) {
      float* wo = p.ws_o + (static_cast<int64_t>(cta) * WS_ROWS + r) * WS_HD;
#pragma unroll
      for (int j = 0; j < 32; ++j) {
        const float2 v2 = make_float2(acc_o[4 * j + 2 * rh] * os, acc_o[4 * j + 2 * rh + 1] * os);
        *reinterpret_cast<float2*>(wo + 8 * j + 2 * q4) = v2;
      }
      // Every (kvh, dh) CTA has its own workspace slot and its combine reads that slot's LSE, so each writes it.
      if (q4 == 0) p.ws_lse[static_cast<int64_t>(cta) * WS_ROWS + r] = lse2;
    } else {
      __nv_bfloat16* o = p.out + static_cast<int64_t>(q0 + t) * p.o_s0
                       + static_cast<int64_t>(kvh * C::GQA + h) * p.o_s1 + dh * WS_HD;
#pragma unroll
      for (int j = 0; j < 32; ++j) {
        const __nv_bfloat162 v2 = __floats2bfloat162_rn(acc_o[4 * j + 2 * rh] * os, acc_o[4 * j + 2 * rh + 1] * os);
        *reinterpret_cast<__nv_bfloat162*>(o + 8 * j + 2 * q4) = v2;
      }
      if (p.lse != nullptr && q4 == 0 && dh == 0) {
        p.lse[static_cast<int64_t>(kvh * C::GQA + h) * p.total_q + q0 + t] = lse2 * 0.6931471805599453f;
      }
    }
  }
}

// Merge the per-split partials (normalized fp32 O + log2 LSE) into bf16 output and LSE.
// grid = (batch, kv_heads * DSPLIT, GQA * max_q), block = 256 = 4 split groups x 64 float4 lanes of head dim.
// Each (kvh, dh) CTA reduces the splits of its own 256-dim half; LSE is written once, by dh == 0.
// Launched as a programmatic dependent of the main kernel: everything up to griddep_wait() overlaps
// the main kernel's tail.
constexpr int COMBINE_MAX_SPLITS = 192;
template <class C>
__global__ void __launch_bounds__(256) attn_combine_kernel(const __grid_constant__ Params p) {
  griddep_wait();  // main kernel finished; its ws_o / ws_lse / eff writes are visible
  const int b = blockIdx.x;
  const int y = blockIdx.y;
  const int kvh = y / C::DSPLIT;
  const int dh = y % C::DSPLIT;
  const int r = blockIdx.z;
  const int q0 = p.cu_seqlens_q[b];
  const int qlen = min(p.cu_seqlens_q[b + 1] - q0, C::MAX_Q);
  if (r >= C::GQA * qlen) return;
  const int eff = p.eff[b];
  if (eff <= 1) return;  // written directly by the main kernel
  const int t = r / C::GQA;
  const int h = r - t * C::GQA;
  const int tid = threadIdx.x;
  const int64_t cta0 = static_cast<int64_t>(b * static_cast<int>(gridDim.y) + y) * p.num_splits;

  __shared__ float s_w[COMBINE_MAX_SPLITS];
  __shared__ float s_stat[2];  // max lse (log2), sum of weights
  __shared__ float4 s_red[3][64];

  if (tid < 32) {
    float mx = -INFINITY;
    for (int s = tid; s < eff; s += 32) mx = fmaxf(mx, __ldcg(p.ws_lse + (cta0 + s) * WS_ROWS + r));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
    float sum = 0.f;
    for (int s = tid; s < eff; s += 32) {
      const float l = __ldcg(p.ws_lse + (cta0 + s) * WS_ROWS + r);
      const float w = (mx == -INFINITY || l == -INFINITY) ? 0.f : ex2(l - mx);
      s_w[s] = w;
      sum += w;
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
    if (tid == 0) {
      s_stat[0] = mx;
      s_stat[1] = sum;
    }
  }
  __syncthreads();
  const float mmax = s_stat[0];
  const float wsum = s_stat[1];

  __nv_bfloat16* o = p.out + static_cast<int64_t>(q0 + t) * p.o_s0
                   + static_cast<int64_t>(kvh * C::GQA + h) * p.o_s1 + dh * WS_HD;
  if (mmax == -INFINITY) {  // every split empty: defined output, not NaN
    if (tid < WS_HD / 2) *reinterpret_cast<__nv_bfloat162*>(o + 2 * tid) = __floats2bfloat162_rn(0.f, 0.f);
    if (p.lse != nullptr && dh == 0 && tid == 0)
      p.lse[static_cast<int64_t>(kvh * C::GQA + h) * p.total_q + q0 + t] = -INFINITY;
    return;
  }
  const int g = tid >> 6;
  const int dq = tid & 63;
  const float* base = p.ws_o + (cta0 * WS_ROWS + r) * WS_HD + 4 * dq;
  float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll 8
  for (int s = g; s < eff; s += 4) {
    const float w = s_w[s];
    const float4 v = __ldcg(reinterpret_cast<const float4*>(base + static_cast<int64_t>(s) * WS_ROWS * WS_HD));
    acc.x += w * v.x;
    acc.y += w * v.y;
    acc.z += w * v.z;
    acc.w += w * v.w;
  }
  if (g > 0) s_red[g - 1][dq] = acc;
  __syncthreads();
  if (g == 0) {
#pragma unroll
    for (int i = 0; i < 3; ++i) {
      const float4 v = s_red[i][dq];
      acc.x += v.x;
      acc.y += v.y;
      acc.z += v.z;
      acc.w += v.w;
    }
    const float inv = 1.f / wsum;
    *reinterpret_cast<__nv_bfloat162*>(o + 4 * dq) = __floats2bfloat162_rn(acc.x * inv, acc.y * inv);
    *reinterpret_cast<__nv_bfloat162*>(o + 4 * dq + 2) = __floats2bfloat162_rn(acc.z * inv, acc.w * inv);
    if (p.lse != nullptr && dh == 0 && dq == 0) {
      p.lse[static_cast<int64_t>(kvh * C::GQA + h) * p.total_q + q0 + t] = (mmax + __log2f(wsum)) * 0.6931471805599453f;
    }
  }
}

}  // namespace attn_decode
}  // namespace green
