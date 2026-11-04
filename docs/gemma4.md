# Gemma-4-31B (FP8 KV) on SM90: attn_decode instances

Model: `RedHatAI/gemma-4-31B-it-FP8-block`. 60 layers, pattern 5 sliding + 1 global.
SGLang passes `RadixAttention(scaling=1.0, logit_cap=0)`; sliding layers pass
`window_size=(sliding_window - 1, 0) = (1023, 0)` and read the SWA KV pool through `swa_page_table`.

| layer group | layers | q heads | kv heads | GQA | head_dim | window_left | KV bytes / token / layer (e4m3) |
|---|---|---|---|---|---|---|---|
| sliding | 50 | 32 | 16 | 2 | 256 | 1023 | 8 KiB, at most 1024 (+q_len) tokens |
| global  | 10 | 32 | 4  | 8 | 512 | -1   | 4 KiB, whole context |

The sm120 path in the fork (`paged_gqa_fp8_sm120`, Triton) does not run on sm90; there FA4 is the
incumbent (`--attention-backend fa4`, fp8 KV).

## Design brief

**Roofline.** Decode is HBM bound. At 3.35 TB/s, one sliding layer costs at most 2.5 us per sequence
(8 MiB, full window), and one global layer costs 1.25 us per 1k tokens of context (4 MiB per 1k tokens).
Arithmetic intensity on real rows is about 2 x 2 x GQA flop/byte (QK + PV): 8 for sliding and 32 for global,
against about 590 flop/B at the FP8 tensor ridge. The design goals are full-bandwidth gathers, no
wave quantization, and no extra HBM passes.

**Reuse.** The Qwen instance (`GQA 6, HD 256`) design stays as is: loader WG, V-transpose WG,
consumer WG, FP8 WGMMA for both GEMMs, and a device-planned split-KV with a PDL combine. The kernel is
templated on `Cfg<GQA, DSPLIT, WINDOW>`. `kv_heads` (grid.y) is a runtime parameter; `window_left` is runtime
but only honoured by `WINDOW = true` instances (see the invariant below).

**Sliding (`Cfg<2,1,true>`).** This is the same tile machinery with `MAX_Q = 64 / 2` (capped at 16 by dispatch).
- Window: query row `t` (absolute position `p = seqlen - qlen + t`) sees keys `[p - W, p]`.
- The CTA skips every tile below `floor((seqlen - qlen - W) / 128)`.
- The device split planner counts only the tiles inside the window, so splits are sized to at most 9 tiles, not to the context.
- The mask adds the lower bound `col < p - W`.
- A row can lose a whole 64-column half to the window, or a split can lose every column of a row. The online softmax already
  treats `-inf` half-tile maxima as "no contribution", and the combine weights an empty split by 0.
  The row's own position is always inside the window, so a row never ends up fully empty.

**Global (`HD 512`).** Candidates:
1. One CTA, two consumer WGs splitting D (K loaded once). This needs 512 threads, which leaves at most 200 registers
   per consumer thread for `acc_o[128] + acc_s[64] + P[16]` plus addressing, so it spills. K 128x512 also needs
   two 64 KiB stages and V 128x512 needs 64 KiB raw + 64 KiB transposed, which does not fit in 227 KiB without
   `TILE_N = 64` (a rewrite of the transpose, the P layout and the PV k-steps).
2. **D split across CTAs (`DSPLIT = 2`, chosen).** Each kv head gets two CTAs. Both compute the full
   `S = Q K^T` over 512 dims (16 k-steps instead of 8). Each owns 256 output dims, so the V gather, V
   transpose and PV path, along with the register budget, are byte-for-byte the HD256 ones.
   - Cost: K is read twice from L2. The two CTAs are adjacent in `blockIdx.y`, so they are co-resident and HBM
     reads K once. QK tensor work doubles, which is still under 3% of the tensor peak.
   - SMEM: Q 32K + K 2 x 64K + V 1 x 32K + VT 32K = 224 KiB (+ barriers and alignment = 230 544 B, which is under 232 448).
   - LSE is identical in both halves, but every `(kvh, dh)` CTA writes its own `ws_lse` slot: the combine
     CTA for half `dh` reads only that half's slot, so a "dh = 0 only" write leaves stale LSE in the other half.
3. Two launches (one per D half). This has the same traffic as (2) but double the launches and no co-residency
   guarantee for the L2 reuse of K. Rejected.

**Parallelism.** Grid = `(splits, kv_heads * DSPLIT, batch)`. The split bound `plan_splits` uses
`units = kv_heads * DSPLIT` and the window-capped context. The workspace stays at
`2*132 + 4*66` CTAs x 64 rows x 256 dims. The bound `batch * units * ceil(264 / (batch * units)) < 264 + batch * units`
holds for all three instances whenever splits > 1.

**Precision.** This is unchanged from the Qwen instance: e4m3 Q/K/V, fp32 accumulation, and P quantized to
e4m3 (x256) per 64-column half. `softmax_scale = 1.0` for Gemma (q/k RMS-normed); the kernel takes it
from the call.

**Fallback.** Any other `(head_dim, GQA)` pair is declined with a reason, as is a right window other than
`-1/0`, a left window on a full-attention instance, `max_seqlen_q > min(64 / GQA, 16)`, or a descale that is
not `[batch, kv_heads]`. All of these run on FA4.

## Invariants (each one was a real bug or race during bring-up)

- **Window tile skip.** `t_lo = max(seqlen - qlen - W, 0) / 128` uses the *first* query row's lower bound
  (the smallest one), so no tile any row needs is skipped. The mask runs on a tile when it reaches past the
  first row's causal bound or starts below `lo_max`, the *last* valid row's lower bound (the largest one).
  Both decisions are CTA-uniform.
- **Per-lane `v_empty` release.** `bar_v_empty` is initialised with 128 arrivals, and every V-transpose lane
  arrives after its own reads of the raw V stage. A single elected arrive after `__syncwarp` let the loader
  overwrite a stage another warp was still reading (racecheck hazard with `V_STAGES = 1` in the HD512 instance).
- **Split planner on window-capped context.** Splits are planned over the tiles in `[t_lo, t_end)`, never the
  full context. Otherwise a long sliding sequence gets mostly-empty splits that the combine has to weight by 0.
- **Window is compile-time (`Cfg::WINDOW`).** A runtime-only window (`|| col < lo` in the mask plus
  `|| n0 < lo_max` in `need_mask`, never taken at `window_left = -1`) slowed the Qwen main kernel by 5-7 %
  (bs 51, min over 60 L2-flushed calls: 8k 328 -> 357 us, 32k 1239 -> 1337 us): it changes the codegen of the
  64-element mask/max loop. Full-attention instances therefore compile exactly the causal-only mask;
  `select` declines a left window on them and the launcher ICHECKs it.
- **K and V gathers share their index math (HD 256).** `load_k_tile` must compute chunk / token with the same
  bit ops as `load_v_tile` (`tid & 15`, `tid >> 4`). The first generic version used signed `tid % KCH` /
  `tid / KCH`; nvcc then could not prove the K and V token indices equal and recomputed the 16 page-table
  addresses for V (Qwen loader loop 403 -> 575 SASS instructions; the loader shares SM sub-partitions with the
  consumer). Together with the runtime window mask this was the whole Qwen regression (SASS 4384 -> 4792
  instructions; after the fix 4408, main kernel back to the original time).

## Verification

- `op_test/sm90/test_attn_decode_fp8.py`: presets QWEN `(256, kv 4, q 24)`, GEMMA_SWA `(256, kv 16, q 32,
  W 1023)` and GEMMA_FULL `(512, kv 4, q 32)`. Gemma cases cover lengths that straddle the window and tile
  edges (1..16384), qlen > 1 row bounds across the window edge, split-KV far beyond the window, DSPLIT boundary
  lengths, descale forms, unit softmax scale and CUDA-graph replay. Tolerances (2e-2 on out, 5e-3 on lse) are unchanged.
- `op_test/sm90/test_attn_extend_fp8.py` (the extend kernel): ragged prefix + extend over page 1/16/64 with fp8 and
  bf16 queries, query-block boundaries, empty queries / sequences, 3000-token prefill, descale forms, out x lse,
  strided / stacked tensors, determinism, CUDA-graph capture replayed over different `cu_seqlens_q` splits, decline
  reasons, and dispatch reaching the kernel instead of the dequant fallback. Same `check_out` tolerance rules.
- HD512 has no FA4 incumbent to compare with. A sequence above tol must stay inside the elementwise e4m3-P
  rounding floor `2^-4 * min(P@|V|, 6/sqrt(3) * sqrt(P^2 @ V^2))`. This is mutation-tested: dropping a split,
  or misweighting it in the combine, fails the floor on long rows.
- compute-sanitizer racecheck, memcheck and synccheck: 0 hazards on all three instances; racecheck 0 hazards on the
  extend kernel (ragged batch, block boundary, full prefill).
- ptxas: 168 registers, 0 spill bytes for each instance.

## HD512 fp8 extend / prefill (`attn_extend`)

FA4 on sm90 asserts `head_dim <= 256` for FP8 (`sglang/kernels/ops/attention/flash_attn/cute/interface.py`,
"SM90 FA4 FP8 supports head_dim <= 256"), so stock `fa4` + fp8 KV crashes on the 10 global layers.
GreenAttention runs their prefill/extend natively on the FP8 cache (`attn_decode.select_extend`, `attn_extend`,
`csrc/sm90/attn_decode_fp8/extend_kernel.cuh`):

- **Work split.** One CTA = (kv head x D-half, block of `EXTEND_BLOCK_Q = 16` query tokens). Two consumer
  warpgroups each own 8 tokens x GQA 8 = one 64-row WGMMA tile and share every K/V tile the CTA gathers, which
  halves the L2 -> SM bytes per query row (at 64 rows per tile the kernel was L2-bound). The two D-half CTAs of a kv
  head both compute `QK^T` over 512 dims and each owns 256 output dims, as in decode. grid.z =
  `total_q / 16 + batch` slots; sequence `b` owns `[cu_q[b] / 16 + b, cu_q[b+1] / 16 + b + 1)`, spare slots exit,
  so the grid depends only on shapes (CUDA-graph safe). Slots run in reverse (longest context first).
- **Warp roles** (384 threads): warps 0-3 producer (cp.async page gathers of K in two 256-dim halves over a
  3-half-slot ring, V gather and the in-register V transpose for the FP8 PV WGMMA, 24 registers); warps 4-7 and
  8-11 the two consumers (decode's QK^T / softmax / PV loop, each with its own causal bounds). Smem: 2 x 32 KB Q,
  3 x 32 KB K half-slots, 32 KB raw V, 32 KB transposed V. No split-KV, no window (`static_assert`).
- **Decode is untouched.** The producer helpers are copies inside `extend_kernel.cuh`; the decode / verify SASS is
  byte-identical to the build before this kernel (Qwen paths unchanged).
- **Instances.** Only `(head_dim 512, GQA 8)`. Every other shape is declined (`"no extend instance"`) and
  runs FA4 natively. Also declined: softcap, non-causal, a left window, `max_seqlen_q` missing, `q_descale`,
  descale not `[batch, kv_heads]`, and more than 65535 slots (`grid.z`); a declined fp8 HD512 call goes to
  `wide_kv` below.
- **Routing.** `dispatch.with_kvcache` names a wide call without an op `extend`. `select_extend` is also
  registered as a second `verify` selector: a short extend (`max_seqlen_q` 9..16) is routed as verify, which
  the global verify instance (max q 8) declines. Consequence: `GREEN_ATTN_DISABLE=extend` does not disable
  the extend kernel for verify-sized calls; disable `verify` as well.
- **Perf** (`bench/bench_attn_extend.py`, H100, ABBA rounds of A B C C B A, L2 flushed, p25 over 60 calls; A =
  `wide_kv` dequant -> FA4 bf16, B = green, C = FA4 bf16 on a bf16 copy of the same cache, so C is a reference the
  fp8 cache cannot actually use). Geomean over 13 Gemma-global shapes: 1.35x vs dequant, 0.93x vs FA4 bf16.

  | bs:prefix:q | dequant us | green us | FA4 bf16 us | green TFLOP/s |
  |---|---|---|---|---|
  | 1:0:8192 | 7005 | 6680 | 6656 | 329 |
  | 1:8192:8192 | 20923 | 19764 | 20152 | 334 |
  | 1:24576:8192 | 50281 | 46185 | 47725 | 333 |
  | 4:4096:2048 | 9486 | 8410 | 8638 | 327 |
  | 32:8192:10 | 11046 | 844 | 1121 | 204 |
  | 4:1024:256 | 478 | 296 | 274 | 261 |
  | 1:0:1024 | 200 | 170 | 142 | 202 |
  | 8:0:512 | 465 | 441 | 315 | 156 |
  | 16:0:1024 | 2654 | 2496 | 2030 | 221 |

  Chunked prefill / extend (the serving path) is at or above FA4 bf16 (1.00-1.03x) and 1.05-1.13x over the
  dequant fallback; many short extends are 13x (FA4 pays the dequant of every prefix). Short full prefills (<= 2k
  tokens) stay 0.71-0.89x of FA4 bf16: few KV tiles per block, so the pipeline fill and the gather dominate.
  Limiter: L2 -> SM gather bandwidth (about 2.3 us of K/V loads vs 1.7 us of tensor time per 128-token tile, from
  the byte / FLOP model). ncu is not available on the test host (`ERR_NVGPUCTRPERM`), so this is not measured.

## HD512 fp8 KV calls green declines (`wide_kv.py`)

FA4's bf16 kernel does run HD512, so the `green` backend handles the calls its kernels decline itself:

- `GreenAttentionBackend.prepare_paged_mha_query` keeps the query in bf16 for `layer.head_dim > 256` (prefill
  and decode), returning the descales unchanged. HD <= 256 layers (Qwen, Gemma sliding) keep the old logic.
- `dispatch.with_kvcache`: every call green does not take whose KV is e4m3 with head_dim > 256 on sm90 goes to
  `wide_kv.attend`. It gathers the pages in the page table, dequantizes them (descale applied in fp32) to a
  compact bf16 paged buffer with an identity page table and calls FA4 in bf16. K and V stay separate tensors:
  FA4 treats `k is v` with head_dim 512 as MLA.
- Memory: about `(2 + 7) B x kv_heads x head_dim` per page-table token (bf16 K held while V goes fp8 -> fp32
  -> bf16), 18 KiB/token for Gemma global. Eager calls above `GROUP_BYTES` (1 GiB) split into request groups
  (one host sync on `cu_seqlens_q`); calls with LSE, masks or aux tensors run as one group.
- Under CUDA-graph capture the fallback raises: the decode page table spans the static max context, so a
  dequantized copy is not graph-safe. Decode of these layers must stay on the green kernel (it accepts every
  Gemma decode shape: no softcap, no window, e4m3 KV, power-of-two page). `GREEN_ATTN_DISABLE=decode` with fp8
  KV on Gemma therefore fails at capture with that message; use `--kv-cache-dtype bfloat16` there.
- Unsupported (raises): `q_descale`, MXFP8 scale factors, mixed K/V dtypes.

Tests: `test/unit/test_wide_kv.py` (CPU: dequant/descale/page remap, request groups, routing, capture refusal)
and `op_test/sm90/test_wide_kv.py` (real FA4 vs an fp32 reference on calls green declines: prefill with prefix,
page 1/64, descale forms, declined decode, request groups).
