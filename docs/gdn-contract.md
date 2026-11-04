# GDN recurrent decode / verify: contract and design brief (GreenAttention A3, SM90)

Scope: Qwen3.8-27B-FP8 hybrid, 48 Gated-DeltaNet layers, production dims
`H = 16` (q/k heads), `HV = 48` (value heads, GQA group `HV/H = 3`), `K = V = 128`,
bf16 activations, `scale = K**-0.5`, fp32 `ssm_state` (bf16 supported), replay ring `L = 32`.
All sglang paths are relative to `/network-volume/rnd-sglang/sglang/python/sglang/`.

## 1. Where the production call goes (read this first)

| Phase | Production path (`--enable-linear-replayssm-spec --linear-replayssm-cache-len 32`, DFLASH 4) | Hook point used by GreenAttention |
|---|---|---|
| decode, `q_len = 1` | `GDNKernelDispatcher.decode_kernel.packed_decode(...)` (`supports_packed_decode=True`, `gdn_backend.py` L1028); `.decode(...)` is the non-packed fallback call (L1061) | `GreenGDNKernel.packed_decode` / `.decode` (A1, op `gdn_decode`) |
| verify, `q_len = 4` | `GDNAttnBackend._replayssm_target_verify` (`gdn_backend.py` L1693-1779) -> `kernels.ops.attention.linear.gdn_replayssm_spec_decode.gdn_replayssm_spec_decode(..., launch_mode="verify")` (imported at call time) | `HookRegistry` AROUND hook on that function (op `gdn_verify`), installed by `green_attention/gdn_recurrent.py` |
| extend / prefill | `FlashInferGDNKernel` | not touched |

`GreenGDNKernel.target_verify` (A1) is the **non-replayssm** form
(`TritonGDNKernel.target_verify` -> `fused_sigmoid_gating_delta_rule_update(disable_state_update=True,
intermediate_states_buffer=..., retrieve_parent_token=...)`, tree/chain verify with per-step state snapshots). The
production configuration never calls it, so the `gdn_verify` selector declines that form with a reason string. The
production verify is served by the replayssm hook (this is a deviation from the plan text, which assumed
`target_verify`; it is forced by the code).

## 2. Per-token math (shared by decode and verify)

For sequence `n`, value head `hv`, key head `h = hv // 3`, token `t`:

```
x     = a[t,hv] + dt_bias[hv]
g     = -exp(A_log[hv]) * (x <= 20 ? log(1 + exp(x)) : x)           # softplus threshold 20, beta 1
beta  = 1 / (1 + exp(-b[t,hv]))
q'    = q[t,h] / sqrt(sum(q^2) + 1e-6) * scale                       # l2norm over K=128 of head h
k'    = k[t,h] / sqrt(sum(k^2) + 1e-6)
S     = exp(g) * S                       # S is [V, K], K contiguous (fp32 or bf16 storage)
u     = beta * (v[t,hv] - S k')          # [V]
S     = S + u (x) k'
o[t,hv] = S q'                           # [V]
```

Fusion: gating, sigmoid, and both l2norms are computed inside the kernels (no pre-pass).

## 3. Decode contract (`gdn_decode`)

Reference implementations: `TritonGDNKernel.packed_decode` (`fla/fused_recurrent.py`, packed kernel) and
`TritonGDNKernel.decode` (`fused_sigmoid_gating_recurrent.py`).

### 3.1 Arguments

| Form | Arguments |
|---|---|
| packed | `mixed_qkv [B, 2*H*K + HV*V]` bf16 (q, k, v concatenated; last stride 1), `a, b [B, HV]` bf16 (any row stride), `A_log [HV]` fp32, `dt_bias [HV]` fp32 or bf16, `scale`, `ssm_states [slots, HV, V, K]` (fp32 or bf16), `cache_indices [B]` int32/int64, `num_v_heads = 48`, `head_v_dim = 128` |
| split (`decode`) | `q, k [1, N, H, K]`, `v [1, N, HV, V]` bf16, `a, b` bf16 `[N, HV]`, `A_log`, `dt_bias`, `ssm_states`, `cache_indices [B]`, `query_start_loc [B+1]` int32 |

### 3.2 Reads / writes

| Tensor | Access |
|---|---|
| `ssm_states[cache_indices[n], hv]` | read once, written once (in place); `[V,K]` row-major, slot stride is taken from `stride(0)` |
| return value | packed: `[B, 1, HV, V]` transposed view like Triton (`out.new_empty(B,1,HV,V).transpose(0,1)`); split: `[1, N, HV, V]`, bf16 |
| everything else | read-only |

### 3.3 Edge semantics (bit-for-bit what the Triton kernels do)

| Case | packed | split |
|---|---|---|
| `cache_indices[n] < 0` (PAD) | output row = 0, state untouched | state read as 0 (`S = 0`), state not written, output still computed from `S = 0` |
| sequence length != 1 (`query_start_loc[n+1]-query_start_loc[n] != 1`) | n/a | output rows zero, state untouched |
| beta | rounded to bf16 (`packed_decode` loads `beta` through a bf16 intermediate) | plain fp32 |

### 3.4 Not accepted (decline reasons are returned to the dispatcher)

replay ring kwargs (`replayssm_d/_k/_g/_write_pos/_force_flush`) non-None, dtype mismatches (activations not bf16, `A_log` not
fp32), `num_v_heads != 48` or `head_v_dim != 128` or `H != 16` or `K != 128`, state not `[slots, 48, 128, 128]` row-major,
`q/k` rows not 16 B aligned, `cu_seqlens` batch not matching `q.shape[1]`.

## 4. Verify contract (`gdn_verify`, replayssm, `gdn_replayssm_spec_circular_kernel`, `IS_FLUSH = False`)

Reference: `gdn_replayssm_spec_decode.py` L52-481 (kernel), L878-1068 (wrapper). Production call: all keyword args,
`launch_mode="verify"`, `null_block_id=-1`, `scale=K**-0.5`, `use_qk_l2norm_in_kernel=True`, `max_spec_len = 4`,
`max_cache_len = 32`, fp32 `checkpoint_state`, `beta_cache = None`.

### 4.1 Arguments

`q, k [T, H, K]`, `v [T, HV, V]`, `a, b [T, HV]` bf16 (`T = sum spec_len`; row stride from `stride(0)`, so the strided
views of a fused qkv buffer are accepted when the inner strides are `K`/`V`/1), `A_log [HV]` fp32, `dt_bias [HV]` fp32 or bf16,
`checkpoint_state [slots, HV, V, K]` (fp32; bf16 variant also implemented),
`d_cache [ring, HV, L, V]` bf16, `k_cache [ring, H, L, K]` bf16, `g_cache [ring, HV, L]` fp32,
`rawv_cache [ring, HV, L, V]` bf16 (D residual), `rawk_cache [ring, H, L, K]` bf16 (k residual), `beta_cache = None`,
`out [T, HV, V]` bf16, `query_start_loc [B+1]` int32, `ssm_state_indices [B]` (checkpoint slot; `<= -1` = PAD),
`replay_indices [B]` (ring slot), `write_pos, cache_base [ring+1]` int32, `is_flush [ring+1]` int8.

### 4.2 Reads / writes (non-flush rows)

| Tensor | Access |
|---|---|
| `checkpoint_state[ssm_state_indices[n], hv]` | **read only** (verify never writes the checkpoint; folding is a separate kernel not in this path) |
| `d_cache[r, hv, (cache_base+c) & (L-1)]`, `c < write_pos` | read (history of committed deltas); `g_cache` same positions read |
| `k_cache[r, h, (cache_base+c) & (L-1)]` | read (history keys, high part only) |
| `out[bos+j, hv]`, `j < spec_len` | written (bf16) |
| `d_cache[r, hv, phys_j]`, `rawv_cache[r, hv, phys_j]` | written: `d_hi = bf16(D_j)`, `rawv = bf16(D_j - d_hi)`, for `phys_j = (cache_base + write_pos + j) & (L-1)`, only if `write_pos + j < L` and `j < spec_len` |
| `g_cache[r, hv, phys_j]` | written (`g_j`) under the same condition |
| `k_cache[r, h, phys_j]`, `rawk_cache[r, h, phys_j]` | written by the program with `hv % 3 == 0` only: `k_store = bf16(k')`, `rawk = bf16(k' - k_store)` |
| `write_pos`, `cache_base`, `is_flush` | read only (advanced later by `commit_gdn_replayssm_*`) |

### 4.3 Row handling

* `ssm_state_indices[n] <= -1`: zero the `out` rows of that sequence, write nothing else.
* `is_flush[replay_indices[n]] != 0`: return immediately, **no output and no ring writes** (the flush kernels own that row).
* `spec_len = query_start_loc[n+1] - query_start_loc[n]` may be 0..4 (ragged batches); `spec_len = 0` writes nothing.
* The ring write is dropped (but `out` is still produced) when `write_pos + j >= L`.

### 4.4 Math (what the kernel computes)

With `spec_len <= 4` tokens, `G = inclusive cumsum(g)`, `kk[i,j] = k'_i . k'_j`, `kq[i,j] = k'_i . q'_j`:

```
A[i,j]   = (i > j) * beta_i * exp(G_i - G_j) * kk[i,j]           T = (I + A)^-1
hw_q[v,j] = S0[v] . q'_j                  hw_k[v,j] = S0[v] . k'_j
history (c < write_pos):  decay_c = exp(sum_all(g_hist) - prefix_c(g_hist)),  total = exp(sum_all(g_hist))
   hw_x[v,j] = total * hw_x[v,j] + sum_c (d_hist[c,v] * decay_c) * (khist_c . x'_j)      x in {q, k}
R[v,j]   = beta_j * (v[j,v] - exp(G_j) * hw_k[v,j])
D[v,i]   = sum_j R[v,j] * T[i,j]
F[i,j]   = (i <= j) * exp(G_j - G_i) * kq[i,j]
O[v,j]   = exp(G_j) * hw_q[v,j] + sum_i D[v,i] * F[i,j]
```

fp32 checkpoint: q', k', khist, d are used in fp32 (khist is the bf16 high part only, no residual) exactly as the Triton
`STATE_FP32` branch. Triton evaluates every dot with `dot_precision = "tf32"` by default; this kernel uses full fp32 FMAs,
so it is **more** accurate than the incumbent: tolerances are therefore checked against an fp64 reference and against Triton.

Not accepted (decline -> `original` Triton launch): `launch_mode != "verify"`, `beta_cache is not None` (exact-raw mode), `rawv/rawk`
missing, `L` not in {16, 32} or `d_cache.shape[2] != L` or `max_spec_len > 4`, non-bf16 activations, `use_qk_l2norm_in_kernel=False`, `null_block_id != -1`,
head dims other than 16/48/128/128, ring tensors not contiguous within a slot, `is_flush` not int8, `out` not `[T, HV, V]` row-major.

## 5. Design brief

```
Problem   : decode  : per (seq, hv) one rank-1 update of S[128x128]: read S once, write S once.
            verify  : per (seq, hv) 4 chained updates, but S is read-only; extra ring traffic.
            Production batch 1..51, 48 layers x 1 step each per token; regime is HBM-bound.
Roofline  : decode bytes = 2 * 128*128*4 = 131 072 B per (seq,hv) fp32 (65 536 B bf16); bs 51 -> 321 MB (160 MB).
              SOL @3.35 TB/s: 96 us (48 us); measured achievable copy r+w ceiling on this (shared) box 2.85-3.03 TB/s.
            verify bytes = 65 536 (S) + 256*write_pos (d) + ~1.5 KB (k hist L2-shared, rings written) per (seq,hv);
              bs 51, write_pos avg 16: ~ 190 MB -> SOL 57 us (hits the same ceiling).
            FLOPs: decode 4 flops/element (2 FMA), AI = 0.5 flop/B -> 0.1% of tensor peak: no tensor core needed.
              verify 16 flop/element on S (8 dot vectors), 32-64 flops/B -> still < 10% of FP32 FFMA peak.
Candidates: decode
              D1 one warp / CTA, R rows each, all state loads issued up front, reduce-scatter of dots (chosen).
              D2 warp-per-row with plain 5-stage shuffle reduction (3x more shuffles, no gain: both HBM bound).
              D3 TMA 1D bulk prefetch of the S tile into smem + 2 CTAs/SM (adds smem round trip; measured
                 equal-or-worse than register-resident loads for a pure streaming op).
            verify
              V1 CTA = (seq, hv), 4 warps, fp32 FFMA on CUDA cores, lane-group of 8 lanes per row, vectors from smem (chosen).
              V2 tensor-core tf32 mma (like Triton tl.dot): needs S through smem, 3xTF32 for accuracy parity, no
                 advantage when memory-bound; Triton's measured 40% HBM utilization is the latency of its serial
                 dot/solve chain, not FLOPs.
              V3 split V across 2 CTAs per (seq,hv) (Triton BV=64): doubles q/k/gating prologue work, helps only wave
                 quantization at tiny batch.
Chosen    : D1 and V1. Both keep S in registers (no smem round trip), because a single streaming pass has no reuse.
            Verify is split in two kernels chained with PDL: verify_prep_kernel (grid B*H, 128 thr, 63 regs, 5760 B
            smem: q/k l2norm, g/beta gating, (I+A)^-1 factors) and verify_kernel (grid B*HV, 128 thr,
            __launch_bounds__(128,2), 218 regs fp32 / 207 bf16, 13440 B smem). The main kernel waits on the PDL
            signal before it reads the prep workspace (79.9 vs 82.5 us at B=32 against the other orders).
Tiling    : decode grid (V/R, B*HV), 32 threads; R = 8 for fp32 state, R = 16 for bf16 state at B >= 32 (else 8).
            verify as above, 4 warps x 32 rows, 2 CTAs/SM.
SMEM      : decode 0. verify: prep 5760 B, main 13440 B per CTA.
Registers : decode R8 64-72, R16 96-119 (R32 ~167, not used). verify prep 63, main 218/207. 0 spills everywhere.
Workspace : one prep->main scratch, 256 sequence slots (~30 MB), allocated by _prewarm before CUDA-graph capture.
            Graph-safe (no host sync, fixed pointers): the replayssm hook also runs inside the captured
            target-verify graph (op_test: test_verify_graph_replay_matches_eager).
Parallelism: decode B*HV*V/R CTAs (bs 51, R8: 39 168 warps-CTAs ~ 148 per SM resident). verify B*HV CTAs (bs 51: 2448, 9.3 waves of 264).
Precision : storage fp32|bf16, all math fp32, `exp`/`log` via fast-path intrinsics only where Triton uses the same.
            Re-association S k' -> e^g (S0 k') changes rounding only (measured state rel err <= 3e-8 fp32).
Regime    : verify runs green only for fp32 state and B >= 16 (VERIFY_MIN_BATCH); bf16-state verify is 0.65-0.72x
            Triton and B <= 8 is ~0.9x, so both decline to Triton. Decode is enabled for all B and both dtypes.
Fallback  : the selectors in `green_attention/gdn_recurrent.py` (`_select_packed`, `_select_split`,
            `_verify_can_use`); declines log their reason once per signature.
```

## 6. Measured (H100, `bench/bench_gdn.py`, CUDA-graph replay, L2 flushed, GPU shared with another tenant at 100 % util)

| op | state | B | Triton µs | Green µs | speedup | % of 3.35 TB/s |
|---|---|---|---|---|---|---|
| decode | fp32 | 1 | 10.1 | 9.2 | 1.10 | 20 % |
| decode | fp32 | 8 | 25.1 | 24.4 | 1.02 | 62 % |
| decode | fp32 | 32 | 75.4 | 75.2 | 1.00 | 80 % |
| decode | fp32 | 51 | 115.1 | 115.4 | 1.00 | 83 % |
| decode | bf16 | 1 | 9.1 | 8.1 | 1.13 | 12 % |
| decode | bf16 | 8 | 15.1 | 16.2 | 0.93 | 47 % |
| decode | bf16 | 32 | 43.0 | 41.8 | 1.03 | 72 % |
| decode | bf16 | 51 | 64.4 | 62.6 | 1.03 | 77 % |
| verify (q=4) | fp32 | 32 | 89.5 | 80.7 | 1.11 | 51 % |
| verify (q=4) | fp32 | 51 | 126.0 | 115.1 | 1.09 | 57 % |

Decode is HBM-bound for both kernels: Triton already streams the state at the same rate, so parity is the
ceiling. A load-only stub reached 2.99 TB/s at B=51 fp32. The plan's 85 % BW target is missed by 2 points under
contention. Verify is bound by latency in its dependency chain at 2 CTAs/SM, not by HBM. ncu is unavailable
(ERR_NVGPUCTRPERM), so the limiter is inferred from stub timings. compute-sanitizer racecheck has not been run on
the GDN kernels. The real-input check (`test_real_dump`) needs a server run with `GREEN_DUMP_GDN=1
--disable-cuda-graph`.

## 7. Flush / commit contract (`gdn_flush`, replayssm circular commit)

### Call sites

The flush is not part of decode/verify. It runs once per verify step, **eager** (outside CUDA graphs), after the
verify kernels, for **all 48 GDN layers at once**:

```
DFlashWorkerV2._update_target_mamba_state_after_verify
  → update_mamba_state_after_mtp_verify        (sglang/kernels/ops/attention/linear/gdn_replayssm_spec_decode.py)
    → commit_gdn_replayssm_spec                (Triton cursor advance; NOT hooked)
    → commit_gdn_replayssm_circular            (hooked target; green replaces the whole call)
      → gdn_replayssm_compact_commit_kernel    (Triton fold; grid (2, B, HV·L))
      → _finish_gdn_replayssm_circular_fold_kernel (Triton cursor reset)
```

All caller arguments are keyword; `_commit_around` requires `kwargs and not args`. The per-layer
`gdn_replayssm_exact_fold_kernel` path and the `IS_FLUSH=True` branch (legacy `launch_mode="flush"`/`"both"`) are
dead in production and not reimplemented.

### Trigger

In the verify kernel the running write position `wp` advances by the accepted length. The commit fires when
`new_wp + 2·4 > 32` (max_cache_len 32, spec 4), i.e. for fp32 state `new_wp ∈ 25..28`. With accept ≈ 2.4 a row
flushes roughly every 11 steps → steady-state fraction ≈ 0.09 rows/step (5 %–10 % measured band).

### Tensors and semantics

| tensor | shape | notes |
|---|---|---|
| `checkpoint_state` | [L, S, HV, V, K] fp32 | slot `state_batch_indices[i]`; read old, write new |
| `d_cache`/`d_residual_cache` | [L, S2, HV, T, V] bf16 | ring, rows `phys = (base + t) mod 32` |
| `k_cache`/`k_residual_cache` | [L, S2, H, T, K] bf16 | ring, physical rows |
| `g_cache` | [L, S2, HV, T] fp32 | log-decay gate per (hv, t) |
| `write_pos`/`cache_base`/`is_flush` | int32/int32/int8 [S2] | cursors; snapshotted then reset |
| `replay_indices`, `accept_lens` | per row | replay slot, accepted count |

Per flushing row (`n_hist = write_pos[replay]`, `mask = t < n_hist`):

- `decay_t = exp(total − prefix_t)` for `t < len`, else 0; `total = Σ g[phys]`.
- `u = (d + dres)·decay`, `hi = bf16(u)`, `lo = bf16(u − hi)` (same split as Triton).
- `delta = Kᵀ·hi + Kᵀ·lo + Kresᵀ·hi + Kresᵀ·lo` (bf16 mma, fp32 accumulate).
- `H_new = H_old · exp(total) + delta`, written to the same slot (read-modify-write in place).
- Cursors are reset exactly as the Triton finish kernel: `cache_base += n_hist (mod 32)`, `write_pos = 0`,
  `is_flush = 0`. Track rows additionally write the OLD state to `mamba_track_indices`; cursor untouched for them.

Bit-exactness: **not bit-exact to Triton** on the state (different MMA/tile order). Measured (fp64 reference):
green rel err 2.17e-6 vs Triton's own 2.17e-6 — identical error floor. Cursors are bit-identical. Deterministic
run-to-run. Idle rows are skipped in the job list; `null_block_id` rows skipped; empty job list is a no-op.

### Measured (`bench/bench_gdn_flush.py`, CUDA-graph replay, L2 flushed, ABBA; neighbor vLLM idle, util 0–17 %)

48-layer scaled, commit path (cursor advance + fold), µs:

| B | rows | Triton | Green | ratio |
|---|---|---|---|---|
| 1 | 0 (idle) | 8.2 | 7.3 | 0.89× |
| 8 | 0 (idle) | 29.1 | 7.3 | 0.25× |
| 32 | 0 (idle) | 114.9 | 7.0 | 0.06× |
| 51 | 0 (idle) | 220.4 | 7.1 | 0.03× |
| 8 | 9 % | 127.9 | 173.9 | 1.36× |
| 32 | 9 % | 407.0 | 529.7 | 1.30× |
| 51 | 5 % | 506.9 | 554.1 | 1.09× |
| 51 | 9 % | 722.1 | 967.3 | 1.34× |
| 51 | all (51) | 6257 | 8420 | 1.35× |

Per flushing row: Triton 118 µs, green 165 µs (1.35–1.4×). The bench's fraction rows are a fixed
`round(frac·B)` flushing rows, so they *are* the realistic per-step cost; at B=51 a step with zero flushing rows
has probability 0.91⁵¹ ≈ 0.8 %, so the idle win almost never applies. Measured directly, green is slower in every
realistic case: B=8 1.36×, B=32 1.30×, B=51 1.09× (5 %) / 1.34× (9 %). A linear model built from the all-rows
case (`t_idle + n·per_row`) predicts parity at B=51 (769 vs 772 µs/step), but it disagrees with these direct
measurements and is not used for the decision. Green idles at ~7 µs regardless of B (one persistent grid);
Triton pays 220 µs at B=51 when nothing flushes, which only helps small, mostly idle batches.

Roofline (B=51, L=4, one commit): 654 MB state r+w + 215 MB ring reads = 1.5 GB; contiguous-copy floor
2.9–3.0 TB/s → **floor ≈ 517 µs**. Triton (6257 µs / 12) = 521 µs/4-layer ≈ 100 % of floor. Green 8420/12 =
702 µs = 73 % of floor (2.12 vs 2.87 TB/s effective). Occupancy is NOT the limiter (grid 132/264/396 CTAs →
694/719/718 µs); slot identity is irrelevant (712 µs); MMA is 2 % busy; bytes-in-flight is adequate (98 KB/SM vs
15 KB Little's-law need). The residual gap is per-warp latency stalls in the 24-tile m-loop (LDSM→HMMA chains,
60 B spill stores in the no-res variant). Triton matches its own floor because each of its 19584 tiny CTAs has a
short, fully-pipelined t-loop; beating it requires restructuring the fold into Triton's shape (one small CTA per
(64 v, hv, layer) slice), which conflicts with the persistent-grid design that gives the idle win.

### Decision

**Not shipped.** At the production flush fraction green is 9–36 % slower than Triton per step, and the flush is
~2 % of a 30–40 ms step either way. The implementation (`green_attention/gdn_flush.py`,
`csrc/sm90/gdn_flush/`, `op_test/sm90/test_gdn_flush.py`, `bench/bench_gdn_flush.py`, `gdn_flush` disable token)
lives on branch `exp/gdn-flush` (27 op tests pass there). Revisit only with Triton's tile shape (one small CTA per
(64 v, hv, layer) slice) plus a cheap idle early-out.
