// tvm_ffi entry points for the SM90 FP8-KV decode / verify attention kernel (see kernel.cuh).
#include <cuda_runtime.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>

#include <cstdint>

#include "sm90/attn_decode_fp8/kernel.cuh"

namespace {

using tvm::ffi::TensorView;
namespace ad = green::attn_decode;

template <bool Q_BF16, bool SPLIT>
void set_smem_attr() {
  cudaError_t err = cudaFuncSetAttribute(ad::attn_decode_kernel<Q_BF16, SPLIT>,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         static_cast<int>(ad::SMEM_ALLOC));
  TVM_FFI_ICHECK(err == cudaSuccess) << "cudaFuncSetAttribute: " << cudaGetErrorString(err);
}

// Opt every kernel instance into the large dynamic smem carve-out. Called once from prewarm.
void init() {
  set_smem_attr<false, false>();
  set_smem_attr<false, true>();
  set_smem_attr<true, false>();
  set_smem_attr<true, true>();
}

template <bool Q_BF16>
void launch_main(const ad::Params& p, int batch, bool split, cudaStream_t stream) {
  const dim3 grid(static_cast<unsigned>(p.num_splits), ad::KV_HEADS, static_cast<unsigned>(batch));
  if (split) {
    ad::attn_decode_kernel<Q_BF16, true><<<grid, ad::THREADS, ad::SMEM_ALLOC, stream>>>(p);
  } else {
    ad::attn_decode_kernel<Q_BF16, false><<<grid, ad::THREADS, ad::SMEM_ALLOC, stream>>>(p);
  }
}

// q: uint8 (e4m3) or int16 (bf16) view [total_q, 24, 256]; k/v: uint8 views [pages, page, 4, 256];
// out: int16 view of bf16 [total_q, 24, 256]; lse: fp32 [24, total_q] (read only if has_lse);
// kdesc/vdesc: fp32 [batch|1, 4] (read only if has_kdesc/has_vdesc, strides honored);
// ws_o: fp32 [cta, 64, 256]; ws_lse: fp32 [cta, 64]; ws_eff: int32 [>= batch].
// num_splits is the grid upper bound; the splits actually used per sequence are planned on device.
void attn_decode(TensorView q, TensorView k, TensorView v, TensorView page_table, TensorView cache_seqlens,
                 TensorView cu_seqlens_q, TensorView kdesc, TensorView vdesc, TensorView out, TensorView lse,
                 TensorView ws_o, TensorView ws_lse, TensorView ws_eff, int64_t q_bf16, int64_t num_splits,
                 int64_t page_shift, int64_t has_lse, int64_t has_kdesc, int64_t has_vdesc, int64_t batch,
                 int64_t max_q, int64_t sm_count, double softmax_scale) {
  ad::Params p;
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
  p.ws_o = static_cast<float*>(ws_o.data_ptr());
  p.ws_lse = static_cast<float*>(ws_lse.data_ptr());
  p.scale_log2 = static_cast<float>(softmax_scale * 1.4426950408889634);
  p.num_splits = static_cast<int>(num_splits);
  p.batch = static_cast<int>(batch);
  p.sm_count = static_cast<int>(sm_count);
  p.eff = static_cast<int*>(ws_eff.data_ptr());

  const DLDevice dev = q.device();
  cudaStream_t stream = static_cast<cudaStream_t>(::TVMFFIEnvGetStream(dev.device_type, dev.device_id));
  const bool split = num_splits > 1;
  if (q_bf16) {
    launch_main<true>(p, static_cast<int>(batch), split, stream);
  } else {
    launch_main<false>(p, static_cast<int>(batch), split, stream);
  }
  if (split) {
    TVM_FFI_ICHECK(num_splits <= ad::COMBINE_MAX_SPLITS) << "num_splits exceeds the combine kernel limit";
    // Programmatic dependent launch: the combine grid is queued while the main grid drains; it blocks in
    // griddepcontrol.wait until the main grid has completed (main triggers at its first instruction).
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(static_cast<unsigned>(batch), ad::KV_HEADS, static_cast<unsigned>(ad::GQA * max_q));
    cfg.blockDim = dim3(256);
    cfg.dynamicSmemBytes = 0;
    cfg.stream = stream;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    cudaLaunchKernelEx(&cfg, ad::attn_combine_kernel, p);
  }
  cudaError_t err = cudaGetLastError();
  TVM_FFI_ICHECK(err == cudaSuccess) << "green attn_decode launch failed: " << cudaGetErrorString(err);
}

}  // namespace

TVM_FFI_DLL_EXPORT_TYPED_FUNC(init, init);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(attn_decode, attn_decode);
