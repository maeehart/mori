// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// MIT License
#pragma once

#include "mori/core/core.hpp"
#include "mori/ops/dispatch_combine/dispatch_combine.hpp"
#include "mori/shmem/shmem.hpp"

namespace mori {
namespace moe {

#define AR_MAX_GPUS 8

/// Arguments for the intra-node all-reduce kernel.
template <typename T>
struct EpAllReduceArgs {
  T* inputBuf;       // [numTokens, hiddenDim] – local MoE output
  T* outputBuf;      // [numTokens, hiddenDim] – result (may alias shmem out buf)
  mori::application::SymmMemObjPtr shmemInpBuf;     // symmetric P2P-accessible input
  mori::application::SymmMemObjPtr shmemOutBuf;      // symmetric P2P-accessible output
  mori::application::SymmMemObjPtr crossDeviceBarrierMemObj;
  uint64_t* crossDeviceBarrierFlag;
  uint32_t* gridBarrier;
  int rank;
  int worldSize;
  int numTokens;
  int hiddenDim;
};

/* ─────────────────── cross-device barrier (reused from combine) ─────────── */
template <typename T>
inline __device__ void AllReduceBarrierKernel(EpAllReduceArgs<T> args,
                                               const uint64_t barrierFlag) {
  int thdId = threadIdx.x;
  int laneId = threadIdx.x & (warpSize - 1);
  int globalThdId = blockIdx.x * blockDim.x + threadIdx.x;

  int warpNum = blockDim.x / warpSize;
  int globalWarpNum = gridDim.x * warpNum;

  if (laneId == 0) atomicAdd(args.gridBarrier, 1);

  if (globalThdId < args.worldSize) {
    shmem::ShmemUint32WaitUntilEquals(args.gridBarrier, globalWarpNum);
    args.gridBarrier[0] = 0;

    __threadfence_system();
    core::AtomicStoreRelaxedSystem(
        args.crossDeviceBarrierMemObj->template GetAs<uint64_t*>(globalThdId) +
            args.rank,
        barrierFlag);
  }

  if (globalThdId == 0) atomicAdd(args.crossDeviceBarrierFlag, 1);

  uint64_t* localBarrierPtr =
      args.crossDeviceBarrierMemObj->template GetAs<uint64_t*>();
  if (thdId < args.worldSize) {
    while (core::AtomicLoadRelaxedSystem(localBarrierPtr + thdId) !=
           barrierFlag) {
    }
  }
  __syncthreads();
}

/* ─────────────────── P2P all-reduce kernel ──────────────────────────────── */
/// Each GPU:
///   1. Copies its local MoE output into a symmetric buffer.
///   2. Cross-device barrier so all peers' data is visible.
///   3. Reads from all peers via P2P (XGMI) and sums with WarpAccum.
///
/// This is equivalent to `EpCombineIntraNodeKernel` but without the
/// token-routing / expert-weighting logic: every GPU sums the same
/// token index across all ranks (simple element-wise SUM all-reduce).
template <typename T>
__global__ void EpAllReduceIntraNodeKernel(EpAllReduceArgs<T> args) {
  const int thdId = threadIdx.x;
  const int warpId = thdId / warpSize;
  const int laneId = thdId & (warpSize - 1);
  const int warpNum = blockDim.x / warpSize;
  const int globalWarpId = blockIdx.x * warpNum + warpId;
  const int globalWarpNum = gridDim.x * warpNum;

  const int myPe = args.rank;
  const int npes = args.worldSize;
  const int numTokens = args.numTokens;
  const int hiddenDim = args.hiddenDim;

  // ── Phase 1: copy local input → symmetric buffer ──────────────────────
  for (int tok = globalWarpId; tok < numTokens; tok += globalWarpNum) {
    core::WarpCopy(
        args.shmemInpBuf->template GetAs<T*>() + tok * hiddenDim,
        args.inputBuf + tok * hiddenDim, hiddenDim);
  }

  // ── Phase 2: cross-device barrier ─────────────────────────────────────
  const uint64_t barrierFlag = args.crossDeviceBarrierFlag[0];
  AllReduceBarrierKernel(args, barrierFlag);

  // ── Phase 3: read from all peers and accumulate ───────────────────────
  // Shared memory for source pointers (one per PE per warp).
  extern __shared__ char sharedMem[];
  T** srcPtrs = reinterpret_cast<T**>(sharedMem) + warpId * AR_MAX_GPUS;

  for (int tok = globalWarpId; tok < numTokens; tok += globalWarpNum) {
    // Build pointer array – one entry per PE.
    if (laneId < npes) {
      srcPtrs[laneId] =
          args.shmemInpBuf->template GetAs<T*>(laneId) + tok * hiddenDim;
    }
    // WarpAccum reads from all `npes` pointers and sums into output.
    core::WarpAccum<T, 4>(args.outputBuf + tok * hiddenDim, srcPtrs, nullptr,
                          npes, hiddenDim);
  }
}

}  // namespace moe
}  // namespace mori
