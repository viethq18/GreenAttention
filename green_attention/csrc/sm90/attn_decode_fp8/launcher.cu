// tvm_ffi entry points for the SM90 FP8-KV decode / verify (kernel.cuh) and extend (extend_kernel.cuh) attention.
// Decode / verify are instantiated for CfgQwen (GQA 6, HD 256), CfgGemmaSliding (GQA 2, HD 256) and
// CfgGemmaGlobal (GQA 8, HD 512); extend for the Gemma-4 global shape only.
#include <cuda_runtime.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>

#include <cstdint>

#include "sm90/attn_decode_fp8/extend_kernel.cuh"

namespace {

using tvm::ffi::TensorView;
namespace ad = green::attn_decode;

template <class C, bool Q_BF16, bool SPLIT>
void set_smem_attr() {
  cudaError_t err = cudaFuncSetAttribute(ad::attn_decode_kernel<C, Q_BF16, SPLIT>,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         static_cast<int>(C::SMEM_ALLOC));
  TVM_FFI_ICHECK(err == cudaSuccess) << "cudaFuncSetAttribute: " << cudaGetErrorString(err);
}

template <bool Q_BF16>
void set_extend_smem_attr() {
  cudaError_t err = cudaFuncSetAttribute(ad::attn_extend_kernel<Q_BF16>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         static_cast<int>(ad::ExtCfg::SMEM_ALLOC));
  TVM_FFI_ICHECK(err == cudaSuccess) << "cudaFuncSetAttribute: " << cudaGetErrorString(err);
}

// Opt every kernel instance into the large dynamic smem carve-out. Called once from prewarm.
void init() {
  set_smem_attr<ad::CfgQwen, false, false>();
  set_smem_attr<ad::CfgQwen, false, true>();
  set_smem_attr<ad::CfgQwen, true, false>();
  set_smem_attr<ad::CfgQwen, true, true>();
  set_smem_attr<ad::CfgGemmaSliding, false, false>();
  set_smem_attr<ad::CfgGemmaSliding, false, true>();
  set_smem_attr<ad::CfgGemmaSliding, true, false>();
  set_smem_attr<ad::CfgGemmaSliding, true, true>();
  set_smem_attr<ad::CfgGemmaGlobal, false, false>();
  set_smem_attr<ad::CfgGemmaGlobal, false, true>();
  set_smem_attr<ad::CfgGemmaGlobal, true, false>();
  set_smem_attr<ad::CfgGemmaGlobal, true, true>();
  set_extend_smem_attr<false>();
  set_extend_smem_attr<true>();
}

// Fields shared by decode and extend. q: uint8 (e4m3) or int16 (bf16) view [total_q, q_heads, head_dim];
// k/v: uint8 views [pages, page, kv_heads, head_dim]; out: int16 view of bf16, same layout as q; lse: fp32
// [q_heads, total_q] (read only if has_lse); kdesc/vdesc: fp32 [batch|1, kv_heads] (read only if
// has_kdesc/has_vdesc, strides honored).
template <class C>
ad::Params make_params(const TensorView& q, const TensorView& k, const TensorView& v, const TensorView& page_table,
                       const TensorView& cache_seqlens, const TensorView& cu_seqlens_q, const TensorView& kdesc,
                       const TensorView& vdesc, const TensorView& out, const TensorView& lse, int64_t page_shift,
                       int64_t has_lse, int64_t has_kdesc, int64_t has_vdesc, int64_t batch, double softmax_scale) {
  TVM_FFI_ICHECK(q.size(2) == C::HD && q.size(1) == k.size(2) * C::GQA)
      << "green attn_decode: shape mismatch for the Cfg<GQA=" << C::GQA << ", HD=" << C::HD << "> instance";
  ad::Params p = {};
  p.q = q.data_ptr();
  p.q_s0 = q.stride(0);
  p.q_s1 = q.stride(1);
  p.k = static_cast<const uint8_t*>(k.data_ptr());
  p.v = static_cast<const uint8_t*>(v.data_ptr());
  p.k_s0 = k.stride(0);
  p.k_s1 = k.stride(1);
  p.k_s2 = k.stride(2);
  p.v_s0 = v.stride(0);
  p.v_s1 = v.stride(1);
  p.v_s2 = v.stride(2);
  p.kv_heads = static_cast<int>(k.size(2));
  p.page_shift = static_cast<int>(page_shift);
  p.page_mask = (1 << page_shift) - 1;
  p.page_table = static_cast<const int*>(page_table.data_ptr());
  p.pt_s0 = page_table.stride(0);
  p.pt_s1 = page_table.stride(1);
  p.cache_seqlens = static_cast<const int*>(cache_seqlens.data_ptr());
  p.cu_seqlens_q = static_cast<const int*>(cu_seqlens_q.data_ptr());
  p.kdesc = has_kdesc ? static_cast<const float*>(kdesc.data_ptr()) : nullptr;
  p.vdesc = has_vdesc ? static_cast<const float*>(vdesc.data_ptr()) : nullptr;
  p.kd_s0 = has_kdesc ? kdesc.stride(0) : 0;
  p.kd_s1 = has_kdesc ? kdesc.stride(1) : 0;
  p.vd_s0 = has_vdesc ? vdesc.stride(0) : 0;
  p.vd_s1 = has_vdesc ? vdesc.stride(1) : 0;
  p.out = static_cast<__nv_bfloat16*>(out.data_ptr());
  p.o_s0 = out.stride(0);
  p.o_s1 = out.stride(1);
  p.lse = has_lse ? static_cast<float*>(lse.data_ptr()) : nullptr;
  p.total_q = q.size(0);
  p.scale_log2 = static_cast<float>(softmax_scale * 1.4426950408889634);
  p.window_left = -1;
  p.num_splits = 1;
  p.batch = static_cast<int>(batch);
  return p;
}

cudaStream_t stream_of(const TensorView& t) {
  const DLDevice dev = t.device();
  return static_cast<cudaStream_t>(::TVMFFIEnvGetStream(dev.device_type, dev.device_id));
}

// Decode / verify. ws_o: fp32 [cta, 64, 256]; ws_lse: fp32 [cta, 64]; ws_eff: int32 [>= batch].
// num_splits is the grid upper bound; the splits actually used per sequence are planned on device.
// window_left: sliding window (query at position p attends keys [p - W, p]); -1 = none.
// Dispatch is by the (head_dim, GQA) instance table; kv_heads is a runtime grid dimension.
template <class C, bool Q_BF16>
void dispatch_combined(const ad::Params& p, int batch, bool split, int64_t max_q, cudaStream_t stream) {
  if (split) {
    ad::attn_decode_kernel<C, Q_BF16, true><<<dim3(p.num_splits, p.kv_heads * C::DSPLIT, batch),
                                           ad::THREADS, C::SMEM_ALLOC, stream>>>(p);
    TVM_FFI_ICHECK(p.num_splits <= ad::COMBINE_MAX_SPLITS) << "num_splits exceeds the combine kernel limit";
    // Programmatic dependent launch: the combine grid is queued while the main grid drains; it blocks in
    // griddepcontrol.wait until the main grid has completed (main triggers at its first instruction).
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(static_cast<unsigned>(batch), static_cast<unsigned>(p.kv_heads * C::DSPLIT),
                       static_cast<unsigned>(C::GQA * max_q));
    cfg.blockDim = dim3(256);
    cfg.dynamicSmemBytes = 0;
    cfg.stream = stream;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    cudaLaunchKernelEx(&cfg, ad::attn_combine_kernel<C>, p);
  } else {
    ad::attn_decode_kernel<C, Q_BF16, false><<<dim3(p.num_splits, p.kv_heads * C::DSPLIT, batch),
                                              ad::THREADS, C::SMEM_ALLOC, stream>>>(p);
  }
}

template <class C>
void attn_decode_impl(const TensorView& q, const TensorView& k, const TensorView& v,
                      const TensorView& page_table, const TensorView& cache_seqlens,
                      const TensorView& cu_seqlens_q, const TensorView& kdesc, const TensorView& vdesc,
                      const TensorView& out, const TensorView& lse, const TensorView& ws_o,
                      const TensorView& ws_lse, const TensorView& ws_eff, int64_t q_bf16, int64_t num_splits,
                      int64_t page_shift, int64_t has_lse, int64_t has_kdesc, int64_t has_vdesc, int64_t batch,
                      int64_t max_q, int64_t sm_count, int64_t window_left, double softmax_scale) {
  TVM_FFI_ICHECK(window_left < 0 || C::WINDOW)
      << "green attn_decode: window_left=" << window_left << " on a full-attention instance (Cfg WINDOW=false)";
  ad::Params p = make_params<C>(q, k, v, page_table, cache_seqlens, cu_seqlens_q, kdesc, vdesc, out, lse, page_shift,
                                has_lse, has_kdesc, has_vdesc, batch, softmax_scale);
  p.ws_o = static_cast<float*>(ws_o.data_ptr());
  p.ws_lse = static_cast<float*>(ws_lse.data_ptr());
  p.window_left = static_cast<int>(window_left);
  p.num_splits = static_cast<int>(num_splits);
  p.sm_count = static_cast<int>(sm_count);
  p.eff = static_cast<int*>(ws_eff.data_ptr());

  cudaStream_t stream = stream_of(q);
  if (q_bf16) {
    dispatch_combined<C, true>(p, static_cast<int>(batch), num_splits > 1, max_q, stream);
  } else {
    dispatch_combined<C, false>(p, static_cast<int>(batch), num_splits > 1, max_q, stream);
  }
  cudaError_t err = cudaGetLastError();
  TVM_FFI_ICHECK(err == cudaSuccess) << "green attn_decode launch failed: " << cudaGetErrorString(err);
}

void attn_decode(TensorView q, TensorView k, TensorView v, TensorView page_table, TensorView cache_seqlens,
                 TensorView cu_seqlens_q, TensorView kdesc, TensorView vdesc, TensorView out, TensorView lse,
                 TensorView ws_o, TensorView ws_lse, TensorView ws_eff, int64_t q_bf16, int64_t num_splits,
                 int64_t page_shift, int64_t has_lse, int64_t has_kdesc, int64_t has_vdesc, int64_t batch,
                 int64_t max_q, int64_t sm_count, int64_t window_left, double softmax_scale) {
  // Instance table: (head_dim, GQA) -> Cfg. kv_heads is a runtime grid dimension; an unknown pair is an
  // ICHECK inside attn_decode_impl (select() has already declined it with a reason string).
  const auto hd = q.size(2);
  const auto gqa = q.size(1) / k.size(2);
  if (hd == ad::CfgGemmaGlobal::HD && gqa == ad::CfgGemmaGlobal::GQA) {
    attn_decode_impl<ad::CfgGemmaGlobal>(q, k, v, page_table, cache_seqlens, cu_seqlens_q, kdesc, vdesc, out, lse,
                                         ws_o, ws_lse, ws_eff, q_bf16, num_splits, page_shift, has_lse, has_kdesc,
                                         has_vdesc, batch, max_q, sm_count, window_left, softmax_scale);
  } else if (hd == ad::CfgGemmaSliding::HD && gqa == ad::CfgGemmaSliding::GQA) {
    attn_decode_impl<ad::CfgGemmaSliding>(q, k, v, page_table, cache_seqlens, cu_seqlens_q, kdesc, vdesc, out, lse,
                                          ws_o, ws_lse, ws_eff, q_bf16, num_splits, page_shift, has_lse, has_kdesc,
                                          has_vdesc, batch, max_q, sm_count, window_left, softmax_scale);
  } else {
    attn_decode_impl<ad::CfgQwen>(q, k, v, page_table, cache_seqlens, cu_seqlens_q, kdesc, vdesc, out, lse, ws_o,
                                  ws_lse, ws_eff, q_bf16, num_splits, page_shift, has_lse, has_kdesc, has_vdesc,
                                  batch, max_q, sm_count, window_left, softmax_scale);
  }
}

// Prefill / extend (causal, bottom-right aligned, no window) over an FP8 cache, Gemma-4 global shape only.
// grid.z = total_q / BLOCK_Q + batch query blocks (see ext_work); the grid depends on shapes only.
void attn_extend(TensorView q, TensorView k, TensorView v, TensorView page_table, TensorView cache_seqlens,
                 TensorView cu_seqlens_q, TensorView kdesc, TensorView vdesc, TensorView out, TensorView lse,
                 int64_t q_bf16, int64_t page_shift, int64_t has_lse, int64_t has_kdesc, int64_t has_vdesc,
                 int64_t batch, double softmax_scale) {
  using E = ad::ExtCfg;
  const ad::Params p = make_params<E::C>(q, k, v, page_table, cache_seqlens, cu_seqlens_q, kdesc, vdesc, out, lse,
                                         page_shift, has_lse, has_kdesc, has_vdesc, batch, softmax_scale);
  const int64_t blocks = q.size(0) / E::BLOCK_Q + batch;
  TVM_FFI_ICHECK(blocks <= 65535) << "green attn_extend: " << blocks << " query blocks exceed grid.z";
  const dim3 grid(1, static_cast<unsigned>(p.kv_heads * E::DSPLIT), static_cast<unsigned>(blocks));
  cudaStream_t stream = stream_of(q);
  if (q_bf16) {
    ad::attn_extend_kernel<true><<<grid, ad::THREADS, E::SMEM_ALLOC, stream>>>(p);
  } else {
    ad::attn_extend_kernel<false><<<grid, ad::THREADS, E::SMEM_ALLOC, stream>>>(p);
  }
  cudaError_t err = cudaGetLastError();
  TVM_FFI_ICHECK(err == cudaSuccess) << "green attn_extend launch failed: " << cudaGetErrorString(err);
}

}  // namespace

TVM_FFI_DLL_EXPORT_TYPED_FUNC(init, init);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(attn_decode, attn_decode);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(attn_extend, attn_extend);
