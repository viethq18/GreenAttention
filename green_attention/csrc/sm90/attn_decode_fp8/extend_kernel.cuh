// SM90 FP8-KV prefill / extend attention for the Gemma-4 global layers (GQA 8, head_dim 512), which FA4 cannot
// run over an FP8 cache on SM90. Causal, bottom-right aligned (query token t of a sequence with qlen new tokens
// and cache_seqlens = seqlen sits at position seqlen - qlen + t), no window, no softcap.
//
// Work split. One CTA = (kv head x D-half dh, block of BLOCK_Q = 16 query tokens). Two consumer warpgroups each
// own 8 consecutive tokens x GQA 8 heads = one 64-row WGMMA tile and share every K / V tile the CTA loads, so the
// gathered (L2 -> SM) bytes per query row are half of what a one-consumer CTA needs; at 64 rows per K/V tile the
// extend kernel was L2-bandwidth bound. As in the decode kernel (kernel.cuh) the two dh CTAs of a kv head both
// compute S = Q K^T over all 512 dims and each owns 256 output dims (V gather / transpose / PV are 256 wide).
//
// Warp roles (384 threads):
//   warps 0-3  producer: cp.async gathers of K (two 256-dim halves per tile, ring of KH_STAGES half-slots) and V
//              (this CTA's 256 dims), and the in-register V transpose into the K-major layout the FP8 PV WGMMA
//              needs (transpose_v). The loader and transposer of the decode kernel are merged into one warpgroup
//              because the smem of the second Q tile leaves no room for a second raw-V stage.
//   warps 4-7  consumer 0: tokens [q0, q0 + 8) of the block
//   warps 8-11 consumer 1: tokens [q0 + 8, q0 + 16)
// Each consumer runs the decode kernel's QK^T / softmax_half / PV loop over every KV tile the block's last token
// sees, with its own causal bounds. A consumer without valid rows (block of <= 8 tokens) only paces the barriers.
//
// Smem map (bytes, 1024-aligned base):
//   sQ   2 x 32K   per consumer: 4 atoms (128 dims each) x 64 rows x 128 B, SW128
//   sK   3 x 32K   K half-slots: 2 atoms x 128 tokens x 128 B, SW128; half j = 2 * tile + h lives in slot j % 3
//   sVr  1 x 32K   raw V tile, token-major 256 B rows, chunk^f(token) swizzle (load_v_tile)
//   sVT  1 x 32K   transposed V (transpose_v)
//   barriers
#pragma once

#include "sm90/attn_decode_fp8/kernel.cuh"

namespace green {
namespace attn_decode {

struct ExtCfg {
  using C = CfgGemmaGlobal;
  static constexpr int GQA = C::GQA;
  static constexpr int HD = C::HD;
  static constexpr int KCH = C::KCH;
  static constexpr int DSPLIT = C::DSPLIT;
  static constexpr int WG_Q = M_TILE / GQA;   // 8 tokens per consumer
  static constexpr int CONSUMERS = 2;
  static constexpr int BLOCK_Q = WG_Q * CONSUMERS;  // 16 tokens per CTA
  static constexpr int KH_STAGES = 3;
  static constexpr int P_REGS = 24;   // producer: 128 x (168 - 24) = 2 x 128 x (240 - 168)
  static constexpr int C_REGS = 240;  // consumer: the decode kernel's 240 (acc_o 128 + acc_s 64 + softmax state)
  static constexpr uint32_t Q_BYTES = M_TILE * HD;      // per consumer
  static constexpr uint32_t KH_BYTES = TILE_N * WS_HD;  // 128 tokens x 256 dims
  static constexpr uint32_t V_BYTES = TILE_N * WS_HD;
  static constexpr uint32_t SQ_OFF = 0;
  static constexpr uint32_t SK_OFF = SQ_OFF + CONSUMERS * Q_BYTES;
  static constexpr uint32_t SV_OFF = SK_OFF + KH_STAGES * KH_BYTES;
  static constexpr uint32_t SVT_OFF = SV_OFF + V_BYTES;
  static constexpr uint32_t BAR_OFF = SVT_OFF + V_BYTES;
  static constexpr uint32_t SMEM_BYTES = BAR_OFF + 128;
  static constexpr uint32_t SMEM_ALLOC = SMEM_BYTES + 1024;  // slack for 1024 B alignment
  static_assert(GQA * WG_Q == M_TILE, "q rows");
  static_assert(DSPLIT == 2 && KCH == 32, "head_dim 512");
  static_assert(SMEM_ALLOC <= SMEM_LIMIT, "smem");
  static_assert(2 * KH_STAGES + 3 <= 16, "barrier area");
  static_assert(P_REGS + 2 * C_REGS <= 3 * 168, "register pool");
};

// Gather the dims [256 h, 256 h + 256) of one 128-token K tile into a half-slot (this thread: 16 tokens x one
// 16 B chunk). Same token / chunk assignment as load_v_tile, SW128 layout of load_k_tile (2 atoms of 128 dims).
// Partially unrolled: the producer warpgroup runs at ExtCfg::P_REGS = 24 registers.
__device__ __forceinline__ void load_k_half(const Params& p, const int* pt, int kvh, int seqlen, int tile, int h,
                                            uint32_t slot, int tid) {
  const int chunk = tid & 15;
  const int trow = tid >> 4;  // 0..7 = SW128 row phase of every token this thread loads
  const uint8_t* head_base = p.k + static_cast<int64_t>(kvh) * p.k_s2 + h * WS_HD + chunk * 16;
  const uint32_t col = slot + (chunk >> 3) * 16384 + (((chunk & 7) ^ trow) << 4);
#pragma unroll 4
  for (int i = 0; i < 16; ++i) {
    const int tok = trow + 8 * i;
    const int n = min(tile * TILE_N + tok, seqlen - 1);  // clamp: masked tail rows read real data
    const int64_t page = pt[static_cast<int64_t>(n >> p.page_shift) * p.pt_s1];
    const uint8_t* src = head_base + page * p.k_s0 + static_cast<int64_t>(n & p.page_mask) * p.k_s1;
    cp_async_16(col + tok * 128, src);
  }
}

// load_v_tile / transpose_v of kernel.cuh (same layouts) with partially unrolled loops for the 24-register
// producer. Kept as copies: refactoring the decode helpers into shared loop bodies changes the decode SASS.
__device__ __forceinline__ void load_v_tile_lean(const Params& p, const int* pt, int kvh, int dh, int seqlen, int tile,
                                                 uint32_t stage, int tid) {
  const int chunk = tid & 15;
  const int trow = tid >> 4;
  const uint8_t* head_base = p.v + static_cast<int64_t>(kvh) * p.v_s2 + dh * WS_HD + chunk * 16;
#pragma unroll 4
  for (int i = 0; i < 16; ++i) {
    const int tok = trow + 8 * i;
    const int n = min(tile * TILE_N + tok, seqlen - 1);
    const int64_t page = pt[static_cast<int64_t>(n >> p.page_shift) * p.pt_s1];
    const uint8_t* src = head_base + page * p.v_s0 + static_cast<int64_t>(n & p.page_mask) * p.v_s1;
    const int f = ((trow >> 1) & 3) | (((i >> 1) & 1) << 2);
    cp_async_16(stage + tok * 256 + ((chunk ^ f) << 4), src);
  }
}
__device__ __forceinline__ void transpose_v_lean(uint32_t vraw, uint32_t vt, int warp_p, int lane) {
  const int q = lane & 3;
  const int kidx = lane >> 2;
  const int t0 = 16 * kidx + 2 * q;
  const int f = ((t0 >> 1) & 3) | (((t0 >> 4) & 1) << 2);
#pragma unroll 1
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

// Block j of sequence b covers query tokens [q0, q0 + qlen), q0 = cu_seqlens_q[b] + BLOCK_Q j, and attends the KV
// prefix [0, seqlen) its last token sees: seqlen = cache_seqlens[b] - (cu_seqlens_q[b + 1] - q0) + qlen. Sequence b
// owns the z range [bs(b), bs(b + 1)), bs(b) = cu_seqlens_q[b] / BLOCK_Q + b (>= ceil(qlen_b / BLOCK_Q) slots;
// surplus slots exit with qlen <= 0), so grid.z = total_q / BLOCK_Q + batch is known on the host (CUDA-graph safe).
// z is walked in reverse so the blocks with the longest prefix are scheduled first.
struct ExtWork {
  int b, seqlen, q0, qlen;
};

__device__ __forceinline__ ExtWork ext_work(const Params& p) {
  constexpr int BQ = ExtCfg::BLOCK_Q;
  const int z = static_cast<int>(gridDim.z) - 1 - static_cast<int>(blockIdx.z);
  int lo = 0, hi = p.batch - 1;  // last b with bs(b) <= z
  while (lo < hi) {
    const int mid = (lo + hi + 1) >> 1;
    if (p.cu_seqlens_q[mid] / BQ + mid <= z) lo = mid; else hi = mid - 1;
  }
  const int c0 = p.cu_seqlens_q[lo];
  const int c1 = p.cu_seqlens_q[lo + 1];
  ExtWork w;
  w.b = lo;
  w.q0 = c0 + BQ * (z - (c0 / BQ + lo));
  w.qlen = min(c1 - w.q0, BQ);
  w.seqlen = p.cache_seqlens[lo] - (c1 - w.q0) + w.qlen;
  return w;
}

template <bool Q_BF16>
__global__ void __launch_bounds__(THREADS, 1) attn_extend_kernel(const __grid_constant__ Params p) {
  using E = ExtCfg;
  extern __shared__ uint8_t smem_raw[];
  const uint32_t smem_base = (smem_u32(smem_raw) + 1023u) & ~1023u;

  const int y = blockIdx.y;
  const int kvh = y / E::DSPLIT;
  const int dh = y % E::DSPLIT;
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;

  const ExtWork wk = ext_work(p);
  if (wk.qlen <= 0) return;  // surplus slot; CTA-uniform, before any barrier
  const int b = wk.b;
  const int seqlen = wk.seqlen;
  const int nt = (seqlen + TILE_N - 1) / TILE_N;  // >= 1: seqlen >= qlen >= 1

  const uint32_t sQ = smem_base + E::SQ_OFF;
  const uint32_t sK = smem_base + E::SK_OFF;
  const uint32_t sV = smem_base + E::SV_OFF;
  const uint32_t sVT = smem_base + E::SVT_OFF;
  const uint32_t bar_k_full = smem_base + E::BAR_OFF;
  const uint32_t bar_k_empty = bar_k_full + 8 * E::KH_STAGES;
  const uint32_t bar_v_full = bar_k_empty + 8 * E::KH_STAGES;
  const uint32_t bar_vt_full = bar_v_full + 8;
  const uint32_t bar_vt_empty = bar_vt_full + 8;
  constexpr int CONSUMER_WARPS = 4 * E::CONSUMERS;

  if (tid == 0) {
    for (int s = 0; s < E::KH_STAGES; ++s) {
      mbar_init(bar_k_full + 8 * s, 128);
      mbar_init(bar_k_empty + 8 * s, CONSUMER_WARPS);
    }
    mbar_init(bar_v_full, 128);
    mbar_init(bar_vt_full, 128);
    mbar_init(bar_vt_empty, CONSUMER_WARPS);
  }
  __syncthreads();

  if (warp < 4) {
    // ============================================================ producer warpgroup
    // Per tile lt: K(lt + 1) half 0 -> transpose V(lt) -> V(lt + 1) -> K(lt + 1) half 1. Half 1 reuses the slot of
    // K(lt) half 0, so it is issued last (after QK(lt)); everything before it only waits on PV(lt - 1).
    reg_dealloc<E::P_REGS>();
    const int* pt = p.page_table + static_cast<int64_t>(b) * p.pt_s0;
    auto issue_k = [&](int j) {  // K half j = 2 * tile + h
      const int s = j % E::KH_STAGES;
      if (j >= E::KH_STAGES) mbar_wait(bar_k_empty + 8 * s, ((j / E::KH_STAGES) - 1) & 1);
      load_k_half(p, pt, kvh, seqlen, j >> 1, j & 1, sK + s * E::KH_BYTES, tid);
      cp_async_mbar_arrive_noinc(bar_k_full + 8 * s);
    };
    issue_k(0);
    issue_k(1);
    load_v_tile_lean(p, pt, kvh, dh, seqlen, 0, sV, tid);
    cp_async_mbar_arrive_noinc(bar_v_full);
#pragma unroll 1
    for (int lt = 0; lt < nt; ++lt) {
      const bool more = lt + 1 < nt;
      if (more) issue_k(2 * lt + 2);
      mbar_wait(bar_v_full, lt & 1);
      if (lt > 0) mbar_wait(bar_vt_empty, (lt - 1) & 1);
      transpose_v_lean(sV, sVT, warp, lane);
      fence_proxy_async();
      mbar_arrive(bar_vt_full);
      named_bar_sync(1, 128);  // every producer thread is done reading raw V(lt)
      if (more) {
        load_v_tile_lean(p, pt, kvh, dh, seqlen, lt + 1, sV, tid);
        cp_async_mbar_arrive_noinc(bar_v_full);
        issue_k(2 * lt + 3);
      }
    }
    return;
  }

  // ============================================================== consumer warpgroups
  reg_alloc<E::C_REGS>();
  const int wg = (warp - 4) >> 2;  // 0..1
  const int cw = (warp - 4) & 3;
  const int ctid = tid - 128 - 128 * wg;
  const int q4 = lane & 3;
  const int g = lane >> 2;
  const int q0 = wk.q0 + E::WG_Q * wg;
  const int qlen = min(max(wk.qlen - E::WG_Q * wg, 0), E::WG_Q);
  const int valid_rows = E::GQA * qlen;
  const uint32_t sQw = sQ + wg * E::Q_BYTES;

  if (valid_rows == 0) {
    // Block of <= 8 tokens: consumer 1 has no rows. Pace the barriers only (never more than one phase ahead:
    // every arrive follows the matching full-barrier wait).
#pragma unroll 1
    for (int lt = 0; lt < nt; ++lt) {
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int j = 2 * lt + h;
        mbar_wait(bar_k_full + 8 * (j % E::KH_STAGES), (j / E::KH_STAGES) & 1);
        __syncwarp();
        if (lane == 0) mbar_arrive(bar_k_empty + 8 * (j % E::KH_STAGES));
      }
      mbar_wait(bar_vt_full, lt & 1);
      __syncwarp();
      if (lane == 0) mbar_arrive(bar_vt_empty);
    }
    return;
  }

  const float kd = p.kdesc ? p.kdesc[static_cast<int64_t>(b) * p.kd_s0 + static_cast<int64_t>(kvh) * p.kd_s1] : 1.f;
  const float vd = p.vdesc ? p.vdesc[static_cast<int64_t>(b) * p.vd_s0 + static_cast<int64_t>(kvh) * p.vd_s1] : 1.f;
  const float scale2 = p.scale_log2 * kd;

  float acc_o[128];
#pragma unroll
  for (int i = 0; i < 128; ++i) acc_o[i] = 0.f;
  // m_top / m_run / l_run: see the decode kernel (softmax_half).
  float m_top[2] = {-INFINITY, -INFINITY};
  float m_run[2] = {-INFINITY, -INFINITY};
  float l_run[2] = {0.f, 0.f};

  const int row0 = 16 * cw + g;
  const int row1 = row0 + 8;
  // Causal upper bound (first key this row does not attend). Padding rows never raise the max (softmax is
  // skipped for inactive warps; inside an active warp they keep seqlen, i.e. see the whole block prefix).
  const int base = seqlen - (wk.qlen - E::WG_Q * wg);  // position of this consumer's first token
  const int lim0 = row0 < valid_rows ? base + row0 / E::GQA + 1 : seqlen;
  const int lim1 = row1 < valid_rows ? base + row1 / E::GQA + 1 : seqlen;
  const bool warp_active = 16 * cw < valid_rows;

  // ---- Q tile -> smem (e4m3, SW128 atoms); padding rows are zero
#pragma unroll
  for (int i = 0; i < E::KCH / 2; ++i) {
    const int u = ctid + 128 * i;
    const int r = u / E::KCH;
    const int ch = u % E::KCH;
    uint32_t w0 = 0, w1 = 0, w2 = 0, w3 = 0;
    if (r < valid_rows) {
      const int t = r / E::GQA;
      const int h = r - t * E::GQA;
      const int64_t off = static_cast<int64_t>(q0 + t) * p.q_s0 + static_cast<int64_t>(kvh * E::GQA + h) * p.q_s1;
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
    const uint32_t dst = sQw + (ch >> 3) * 8192 + r * 128 + (((ch & 7) ^ (r & 7)) << 4);
    asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};\n" ::"r"(dst), "r"(w0), "r"(w1), "r"(w2), "r"(w3)
                 : "memory");
  }
  fence_proxy_async();
  named_bar_sync(2 + wg, 128);

  const uint64_t dq = make_desc_sw128(sQw);
  float acc_s[64];
#pragma unroll 1
  for (int lt = 0; lt < nt; ++lt) {
    const int n0 = lt * TILE_N;

    // ---- S = Q K^T (FP8 WGMMA, SS), one K half-slot (8 k-steps of 32 dims) at a time
    wgmma_fence();
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int j = 2 * lt + h;
      const int s = j % E::KH_STAGES;
      mbar_wait(bar_k_full + 8 * s, (j / E::KH_STAGES) & 1);
      fence_proxy_async();
      const uint64_t dk = make_desc_sw128(sK + s * E::KH_BYTES);
#pragma unroll
      for (int kk = 0; kk < 8; ++kk) {
        const int ka = 8 * h + kk;  // k-step over all 512 dims
        const uint64_t da = dq + (((ka >> 2) * 8192 + (ka & 3) * 32) >> 4);
        const uint64_t db = dk + ((((kk >> 2) & 1) * 16384 + (kk & 3) * 32) >> 4);
        mma_qk(da, db, acc_s, ka > 0);
      }
    }
    wgmma_commit();
    wgmma_wait<0>();
#pragma unroll
    for (int i = 0; i < 64; ++i) fence_operand(acc_s[i]);
    __syncwarp();
    if (lane == 0) {
      mbar_arrive(bar_k_empty + 8 * ((2 * lt) % E::KH_STAGES));
      mbar_arrive(bar_k_empty + 8 * ((2 * lt + 1) % E::KH_STAGES));
    }

    // ---- online softmax in two 64-column halves, P -> e4m3 (x256); see the decode kernel
    uint32_t preg[16];
    float al1_0 = 1.f, al1_1 = 1.f;
    float mxh[2][2];
    if (warp_active) {
      const bool need_mask = n0 + TILE_N > base + 1;  // the tile reaches past this consumer's first row bound
      mxh[0][0] = mxh[0][1] = mxh[1][0] = mxh[1][1] = -INFINITY;
#pragma unroll
      for (int i = 0; i < 64; ++i) {
        float x = acc_s[i] * scale2;
        if (need_mask) {
          const int col = n0 + 8 * (i >> 2) + 2 * q4 + (i & 1);
          if (col >= (((i >> 1) & 1) ? lim1 : lim0)) x = -INFINITY;
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
    mbar_wait(bar_vt_full, lt & 1);
    const uint64_t dv = make_desc_sw128(sVT);
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
    if (lane == 0) mbar_arrive(bar_vt_empty);
  }

  // ---- epilogue: bf16 out, natural-log LSE (written by dh == 0)
  float l0 = l_run[0], l1 = l_run[1];
  l0 += __shfl_xor_sync(0xffffffffu, l0, 1);
  l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
  l1 += __shfl_xor_sync(0xffffffffu, l1, 1);
  l1 += __shfl_xor_sync(0xffffffffu, l1, 2);
  const float os0 = l0 > 0.f ? vd * (1.f / 256.f) / l0 : 0.f;
  const float os1 = l1 > 0.f ? vd * (1.f / 256.f) / l1 : 0.f;
  const float lse2_0 = l0 > 0.f ? m_run[0] + __log2f(l0) : -INFINITY;
  const float lse2_1 = l1 > 0.f ? m_run[1] + __log2f(l1) : -INFINITY;
#pragma unroll
  for (int rh = 0; rh < 2; ++rh) {
    const int r = rh ? row1 : row0;
    if (r >= valid_rows) continue;
    const int t = r / E::GQA;
    const int h = r - t * E::GQA;
    const float os = rh ? os1 : os0;
    __nv_bfloat16* o = p.out + static_cast<int64_t>(q0 + t) * p.o_s0 + static_cast<int64_t>(kvh * E::GQA + h) * p.o_s1
                     + dh * WS_HD;
#pragma unroll
    for (int j = 0; j < 32; ++j) {
      const __nv_bfloat162 v2 = __floats2bfloat162_rn(acc_o[4 * j + 2 * rh] * os, acc_o[4 * j + 2 * rh + 1] * os);
      *reinterpret_cast<__nv_bfloat162*>(o + 8 * j + 2 * q4) = v2;
    }
    if (p.lse != nullptr && q4 == 0 && dh == 0) {
      p.lse[static_cast<int64_t>(kvh * E::GQA + h) * p.total_q + q0 + t] = (rh ? lse2_1 : lse2_0) * 0.6931471805599453f;
    }
  }
}

}  // namespace attn_decode
}  // namespace green
