// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// MIT License
#include "mori/ops/allreduce/allreduce.hpp"

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include "mori/core/core.hpp"
#include "mori/shmem/shmem.hpp"
#include "mori/utils/hip_helper.hpp"
#include "mori/utils/mori_log.hpp"
#include "src/ops/allreduce/allreduce_intranode.hpp"

namespace mori {
namespace allreduce {

using namespace mori::application;
using namespace mori::shmem;

static SymmMemObjPtr AllocSymmBuf(size_t size) {
  void* buf = ShmemExtMallocWithFlags(size, hipDeviceMallocUncached);
  HIP_RUNTIME_CHECK(hipMemset(buf, 0, size));
  SymmMemObjPtr obj = ShmemQueryMemObjPtr(buf);
  assert(obj.IsValid());
  return obj;
}

MoriAllReduceHandle::MoriAllReduceHandle(int rank_, int worldSize_, int maxNumTokens_,
                                         int hiddenDim_)
    : rank(rank_), worldSize(worldSize_), maxNumTokens(maxNumTokens_), hiddenDim(hiddenDim_) {
  // Input staging buffer: each PE stores its own [maxNumTokens, hiddenDim] bf16.
  // Peers read from this via XGMI P2P.
  size_t bufSize = static_cast<size_t>(maxNumTokens) * hiddenDim * sizeof(hip_bfloat16);
  shmemInpBuf = AllocSymmBuf(bufSize);
  shmemOutBuf = AllocSymmBuf(bufSize);

  // Cross-device barrier: worldSize uint64 per PE, arranged as [worldSize] matrix.
  size_t barrierSize = worldSize * sizeof(uint64_t);
  crossDeviceBarrierMemObj = AllocSymmBuf(barrierSize);

  // Barrier flag counter (device memory, not symmetric).
  HIP_RUNTIME_CHECK(hipMalloc(&crossDeviceBarrierFlag, sizeof(uint64_t)));
  HIP_RUNTIME_CHECK(hipMemset(crossDeviceBarrierFlag, 0, sizeof(uint64_t)));

  // Grid barrier counter (device memory).
  HIP_RUNTIME_CHECK(hipMalloc(&gridBarrier, sizeof(uint32_t)));
  HIP_RUNTIME_CHECK(hipMemset(gridBarrier, 0, sizeof(uint32_t)));

  multiProcessorCount = GetCurDeviceMultiProcessorCount();

  MORI_OPS_INFO("MoriAllReduceHandle: rank=%d, worldSize=%d, maxTokens=%d, hidden=%d, "
                "inpBuf=%zuB, CUs=%d",
                rank, worldSize, maxNumTokens, hiddenDim, bufSize, multiProcessorCount);
}

MoriAllReduceHandle::~MoriAllReduceHandle() {
  if (shmemInpBuf.IsValid()) ShmemFree(shmemInpBuf->localPtr);
  if (shmemOutBuf.IsValid()) ShmemFree(shmemOutBuf->localPtr);
  if (crossDeviceBarrierMemObj.IsValid()) ShmemFree(crossDeviceBarrierMemObj->localPtr);
  if (crossDeviceBarrierFlag) (void)hipFree(crossDeviceBarrierFlag);
  if (gridBarrier) (void)hipFree(gridBarrier);
}

void MoriAllReduceHandle::Launch(void* input, hipDataType inputType, int numTokens, int blockNum,
                                 int warpPerBlock, hipStream_t stream) {
  const int actualWarpPerBlock = (warpPerBlock <= 0) ? 16 : warpPerBlock;
  const int actualBlockNum = (blockNum <= 0) ? multiProcessorCount : blockNum;
  dim3 grid(actualBlockNum);
  dim3 block(64 * actualWarpPerBlock);  // warpSize=64 on AMD

  // Shared memory: one pointer per PE per warp.
  constexpr int kMaxGpus = 8;
  size_t sharedMemSize = actualWarpPerBlock * kMaxGpus * sizeof(void*);

  // Helper to build args and launch for a specific type.
  auto fillArgs = [&](auto* dummy) {
    using T = std::remove_pointer_t<decltype(dummy)>;
    moe::EpAllReduceArgs<T> args;
    args.inputBuf = reinterpret_cast<T*>(input);
    args.outputBuf = shmemOutBuf->GetAs<T*>();
    args.shmemInpBuf = shmemInpBuf;
    args.shmemOutBuf = shmemOutBuf;
    args.crossDeviceBarrierMemObj = crossDeviceBarrierMemObj;
    args.crossDeviceBarrierFlag = crossDeviceBarrierFlag;
    args.gridBarrier = gridBarrier;
    args.rank = rank;
    args.worldSize = worldSize;
    args.numTokens = numTokens;
    args.hiddenDim = hiddenDim;
    moe::EpAllReduceIntraNodeKernel<T><<<grid, block, sharedMemSize, stream>>>(args);
  };

  switch (inputType) {
    case HIP_R_16BF:
      fillArgs(static_cast<hip_bfloat16*>(nullptr));
      break;
    case HIP_R_32F:
      fillArgs(static_cast<float*>(nullptr));
      break;
    default:
      assert(false && "Unsupported input type for MoriAllReduceHandle::Launch");
      break;
  }
}

}  // namespace allreduce
}  // namespace mori
