// SPDX-License-Identifier: Apache-2.0
// tvm_ffi entry point for the SM90 Replay-SSM commit (flush). See flush.cuh.
#include <cuda_runtime.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>

#include <cstdint>

#include "sm90/gdn_flush/flush.cuh"

namespace gf = green::gdn_flush;
using tvm::ffi::TensorView;

namespace {

template <bool HAS_RES>
void launch_fold(const gf::Params& p, int grid, cudaStream_t stream) {
  static bool configured = false;
  const size_t smem = sizeof(gf::Smem);
  if (!configured) {
    cudaFuncSetAttribute(gf::fold_kernel<HAS_RES>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         static_cast<int>(smem));
    configured = true;
  }
  gf::fold_kernel<HAS_RES><<<grid, gf::kThreads, smem, stream>>>(p);
}

}  // namespace

// state: fp32 [L, S, HV, V, K]; d / dres: bf16 [L, S2, HV, T, V]; k / kres: bf16 [L, S2, H, T, K];
// g: fp32 [L, S2, HV, T]; state_idx / replay_idx / accept / track_idx / track_step: int32|int64 [rows]
// (absent optional tensors are passed as any valid tensor with the has_* flag cleared);
// write_pos / cache_base: int32 [S2]; is_flush: int8|bool [S2]; jobs: int32 [kMaxRows * 8];
// counters: int32 [2]. Each stride argument is in elements.
void gdn_flush(TensorView state, TensorView d, TensorView dres, TensorView k, TensorView kres, TensorView g,
               TensorView state_idx, TensorView replay_idx, TensorView accept, TensorView track_idx,
               TensorView track_step, TensorView write_pos, TensorView cache_base, TensorView is_flush,
               TensorView jobs, TensorView counters, int64_t state_idx_stride, int64_t replay_idx_stride,
               int64_t accept_stride, int64_t track_idx_stride, int64_t track_step_stride, int64_t has_res,
               int64_t has_track, int64_t layers, int64_t rows, int64_t null_id, int64_t grid) {
  TVM_FFI_ICHECK(rows >= 0 && rows <= gf::kMaxRows) << "green gdn_flush: rows " << rows;
  gf::Params p;
  p.state = static_cast<float*>(state.data_ptr());
  p.state_ls = state.stride(0);
  p.state_ss = state.stride(1);
  p.d = static_cast<const __nv_bfloat16*>(d.data_ptr());
  p.d_ls = d.stride(0);
  p.d_ss = d.stride(1);
  p.dres = static_cast<const __nv_bfloat16*>(dres.data_ptr());
  p.dres_ls = dres.stride(0);
  p.dres_ss = dres.stride(1);
  p.k = static_cast<const __nv_bfloat16*>(k.data_ptr());
  p.k_ls = k.stride(0);
  p.k_ss = k.stride(1);
  p.kres = static_cast<const __nv_bfloat16*>(kres.data_ptr());
  p.kres_ls = kres.stride(0);
  p.kres_ss = kres.stride(1);
  p.g = static_cast<const float*>(g.data_ptr());
  p.g_ls = g.stride(0);
  p.g_ss = g.stride(1);
  p.layers = static_cast<int>(layers);
  p.state_idx = state_idx.data_ptr();
  p.replay_idx = replay_idx.data_ptr();
  p.accept = accept.data_ptr();
  p.track_idx = track_idx.data_ptr();
  p.track_step = track_step.data_ptr();
  p.write_pos = static_cast<int*>(write_pos.data_ptr());
  p.cache_base = static_cast<int*>(cache_base.data_ptr());
  p.is_flush = static_cast<int8_t*>(is_flush.data_ptr());
  p.state_idx_stride = state_idx_stride;
  p.replay_idx_stride = replay_idx_stride;
  p.accept_stride = accept_stride;
  p.track_idx_stride = track_idx_stride;
  p.track_step_stride = track_step_stride;
  p.state_idx_bits = state_idx.dtype().bits;
  p.replay_idx_bits = replay_idx.dtype().bits;
  p.accept_bits = accept.dtype().bits;
  p.track_idx_bits = track_idx.dtype().bits;
  p.track_step_bits = track_step.dtype().bits;
  p.has_track = static_cast<int>(has_track);
  p.rows = static_cast<int>(rows);
  p.null_id = static_cast<int>(null_id);
  p.jobs = static_cast<gf::Job*>(jobs.data_ptr());
  p.counters = static_cast<int*>(counters.data_ptr());

  const DLDevice dev = state.device();
  cudaStream_t stream = static_cast<cudaStream_t>(::TVMFFIEnvGetStream(dev.device_type, dev.device_id));
  gf::prep_kernel<<<1, 256, 0, stream>>>(p);
  if (has_res) launch_fold<true>(p, static_cast<int>(grid), stream);
  else launch_fold<false>(p, static_cast<int>(grid), stream);
  cudaError_t err = cudaGetLastError();
  TVM_FFI_ICHECK(err == cudaSuccess) << "green gdn_flush launch failed: " << cudaGetErrorString(err);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(gdn_flush, gdn_flush);
