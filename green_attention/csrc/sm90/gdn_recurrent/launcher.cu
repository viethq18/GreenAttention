// SPDX-License-Identifier: Apache-2.0
// tvm_ffi entry points for the GreenAttention SM90 GDN recurrent kernels.
#include <cuda_runtime.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>

#include <cstdint>

#include "sm90/gdn_recurrent/kernel.cuh"
#include "sm90/gdn_recurrent/verify.cuh"

namespace gdn = green::gdn;
using tvm::ffi::TensorView;

namespace {

template <typename ST, int R, bool PACKED>
void launch_decode(const gdn::DecodeParams& p, int batch, cudaStream_t stream) {
  const dim3 grid(gdn::kV / R, static_cast<unsigned>(batch) * gdn::kHV);
  gdn::decode_kernel<ST, R, PACKED><<<grid, 32, 0, stream>>>(p);
}

template <typename ST>
void launch_verify(const gdn::VerifyParams& p, int batch, cudaStream_t stream) {
  constexpr int NST = 2;
  constexpr int smem = gdn::verify_smem_bytes<ST, NST>();
  gdn::verify_prep_kernel<ST><<<static_cast<unsigned>(batch) * gdn::kH, 128, 0, stream>>>(p);
  static const bool smem_ok = [] {
    return cudaFuncSetAttribute(gdn::verify_kernel<ST, NST>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                smem) == cudaSuccess;
  }();
  TVM_FFI_ICHECK(smem_ok) << "green gdn_verify: cannot opt in to " << smem << " B dynamic smem";
  int dev = 0, num_sms = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev);
  const int ntiles = batch * gdn::kHV;
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(static_cast<unsigned>(ntiles < num_sms ? ntiles : num_sms));
  cfg.blockDim = dim3(160);  // 4 consumer warps (state row = thread) + 1 producer warp
  cfg.dynamicSmemBytes = smem;
  cfg.stream = stream;
  cudaLaunchAttribute attr;
  attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr.val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = &attr;
  cfg.numAttrs = 1;
  cudaLaunchKernelEx(&cfg, gdn::verify_kernel<ST, NST>, p, ntiles, batch);
}

template <typename ST, bool PACKED>
void launch_decode_rows(const gdn::DecodeParams& p, int batch, int rows, cudaStream_t stream) {
  switch (rows) {
    case 4: launch_decode<ST, 4, PACKED>(p, batch, stream); break;
    case 8: launch_decode<ST, 8, PACKED>(p, batch, stream); break;
    case 16: launch_decode<ST, 16, PACKED>(p, batch, stream); break;
    case 32: launch_decode<ST, 32, PACKED>(p, batch, stream); break;
    default: TVM_FFI_ICHECK(false) << "green gdn_decode: unsupported rows-per-CTA " << rows;
  }
}

}  // namespace

// q/k/v: bf16 buffers; `k_off`/`v_off` are element offsets added to k/v data_ptr (the packed form
// slices one buffer three ways) and `*_ts` are token strides in elements.
// a/b: bf16 [tokens, HV]; A_log: fp32 [HV]; dt_bias: fp32|bf16 [HV];
// cache_idx: int32|int64 [B]; cu_seqlens: int32 [B+1] when has_cu, otherwise ignored;
// state: fp32|bf16 [slots, HV, V, K]; out: bf16 [B, HV, V] (contiguous).
void gdn_decode(TensorView q, TensorView k, TensorView v, TensorView a, TensorView b, TensorView A_log,
                TensorView dt_bias, TensorView cache_idx, TensorView cu_seqlens, TensorView state,
                TensorView out, int64_t q_ts, int64_t k_ts, int64_t v_ts, int64_t k_off, int64_t v_off,
                int64_t a_ts, int64_t b_ts, int64_t has_cu, int64_t packed, int64_t rows, int64_t batch,
                double scale) {
  gdn::DecodeParams p;
  p.q = static_cast<const __nv_bfloat16*>(q.data_ptr());
  p.k = static_cast<const __nv_bfloat16*>(k.data_ptr()) + k_off;
  p.v = static_cast<const __nv_bfloat16*>(v.data_ptr()) + v_off;
  p.q_ts = q_ts;
  p.k_ts = k_ts;
  p.v_ts = v_ts;
  p.a = static_cast<const __nv_bfloat16*>(a.data_ptr());
  p.b = static_cast<const __nv_bfloat16*>(b.data_ptr());
  p.a_ts = a_ts;
  p.b_ts = b_ts;
  p.A_log = static_cast<const float*>(A_log.data_ptr());
  p.dt_bias = dt_bias.data_ptr();
  p.dt_bias_bf16 = dt_bias.dtype().code == kDLBfloat ? 1 : 0;
  p.cache_idx = cache_idx.data_ptr();
  p.idx64 = cache_idx.dtype().bits == 64 ? 1 : 0;
  p.cu_seqlens = has_cu ? static_cast<const int*>(cu_seqlens.data_ptr()) : nullptr;
  p.state = state.data_ptr();
  p.state_slot_stride = state.stride(0);
  p.out = static_cast<__nv_bfloat16*>(out.data_ptr());
  p.scale = static_cast<float>(scale);

  const bool bf16_state = state.dtype().code == kDLBfloat;
  const DLDevice dev = q.device();
  cudaStream_t stream = static_cast<cudaStream_t>(::TVMFFIEnvGetStream(dev.device_type, dev.device_id));
  const int B = static_cast<int>(batch);
  const int r = static_cast<int>(rows);
  if (bf16_state) {
    if (packed) launch_decode_rows<__nv_bfloat16, true>(p, B, r, stream);
    else launch_decode_rows<__nv_bfloat16, false>(p, B, r, stream);
  } else {
    if (packed) launch_decode_rows<float, true>(p, B, r, stream);
    else launch_decode_rows<float, false>(p, B, r, stream);
  }
  cudaError_t err = cudaGetLastError();
  TVM_FFI_ICHECK(err == cudaSuccess) << "green gdn_decode launch failed: " << cudaGetErrorString(err);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(gdn_decode, gdn_decode);

// Replay-SSM verify (see verify.cuh).  state/ring pointers are taken as-is; per-tensor slot strides
// come from `stride(0)`.  state_idx/replay_idx: int32|int64 [B]; write_pos/cache_base: int32;
// is_flush: int8; qsl: int32 [B+1]; token strides are in elements.
void gdn_verify(TensorView q, TensorView k, TensorView v, TensorView a, TensorView b, TensorView A_log,
                TensorView dt_bias, TensorView state, TensorView d_cache, TensorView rawv_cache,
                TensorView k_cache, TensorView rawk_cache, TensorView g_cache, TensorView state_idx,
                TensorView replay_idx, TensorView qsl, TensorView write_pos, TensorView cache_base,
                TensorView is_flush, TensorView out, TensorView ws_vec, TensorView ws_coef, TensorView ws_sc,
                int64_t q_ts, int64_t k_ts, int64_t v_ts, int64_t a_ts, int64_t b_ts, int64_t o_ts,
                int64_t ring_len, int64_t batch, double scale) {
  gdn::VerifyParams p;
  p.q = static_cast<const __nv_bfloat16*>(q.data_ptr());
  p.k = static_cast<const __nv_bfloat16*>(k.data_ptr());
  p.v = static_cast<const __nv_bfloat16*>(v.data_ptr());
  p.q_ts = q_ts;
  p.k_ts = k_ts;
  p.v_ts = v_ts;
  p.a = static_cast<const __nv_bfloat16*>(a.data_ptr());
  p.b = static_cast<const __nv_bfloat16*>(b.data_ptr());
  p.a_ts = a_ts;
  p.b_ts = b_ts;
  p.A_log = static_cast<const float*>(A_log.data_ptr());
  p.dt_bias = dt_bias.data_ptr();
  p.dt_bias_bf16 = dt_bias.dtype().code == kDLBfloat ? 1 : 0;
  p.state = state.data_ptr();
  p.state_ss = state.stride(0);
  p.d_cache = static_cast<__nv_bfloat16*>(d_cache.data_ptr());
  p.rawv_cache = static_cast<__nv_bfloat16*>(rawv_cache.data_ptr());
  p.k_cache = static_cast<__nv_bfloat16*>(k_cache.data_ptr());
  p.rawk_cache = static_cast<__nv_bfloat16*>(rawk_cache.data_ptr());
  p.g_cache = static_cast<float*>(g_cache.data_ptr());
  p.d_ss = d_cache.stride(0);
  p.rawv_ss = rawv_cache.stride(0);
  p.k_ss = k_cache.stride(0);
  p.rawk_ss = rawk_cache.stride(0);
  p.g_ss = g_cache.stride(0);
  p.state_idx = state_idx.data_ptr();
  p.replay_idx = replay_idx.data_ptr();
  p.sidx64 = state_idx.dtype().bits == 64 ? 1 : 0;
  p.ridx64 = replay_idx.dtype().bits == 64 ? 1 : 0;
  p.qsl = static_cast<const int*>(qsl.data_ptr());
  p.write_pos = static_cast<const int*>(write_pos.data_ptr());
  p.cache_base = static_cast<const int*>(cache_base.data_ptr());
  p.is_flush = static_cast<const int8_t*>(is_flush.data_ptr());
  p.out = static_cast<__nv_bfloat16*>(out.data_ptr());
  p.o_ts = o_ts;
  p.scale = static_cast<float>(scale);
  p.L = static_cast<int>(ring_len);

  const DLDevice dev = q.device();
  cudaStream_t stream = static_cast<cudaStream_t>(::TVMFFIEnvGetStream(dev.device_type, dev.device_id));
  p.ws_vec = static_cast<float*>(ws_vec.data_ptr());
  p.ws_coef = static_cast<float*>(ws_coef.data_ptr());
  p.ws_sc = static_cast<float*>(ws_sc.data_ptr());
  if (state.dtype().code == kDLBfloat) {
    launch_verify<__nv_bfloat16>(p, static_cast<int>(batch), stream);
  } else {
    launch_verify<float>(p, static_cast<int>(batch), stream);
  }
  cudaError_t err = cudaGetLastError();
  TVM_FFI_ICHECK(err == cudaSuccess) << "green gdn_verify launch failed: " << cudaGetErrorString(err);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(gdn_verify, gdn_verify);
