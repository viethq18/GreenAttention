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
