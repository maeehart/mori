// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// MIT License
#pragma once

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

#include "mori/application/application.hpp"
#include "mori/utils/data_types.hpp"

namespace mori {
namespace allreduce {

/// Lightweight handle for P2P intra-node all-reduce.
///
/// Allocates only the minimum symmetric-memory buffers needed:
///   - input staging buffer (P2P-accessible by all peers)
///   - output buffer
///   - cross-device barrier flags
///   - grid barrier counter
///
/// Unlike EpDispatchCombineHandle, this does NOT allocate dispatch/combine
/// routing maps, weight buffers, or staging buffers.
class MoriAllReduceHandle {
 public:
  MoriAllReduceHandle(int rank, int worldSize, int maxNumTokens, int hiddenDim);
  ~MoriAllReduceHandle();

  /// Launch the P2P all-reduce kernel.
  /// @param input     [numTokens, hiddenDim] bf16 tensor (device ptr).
  /// @param inputType HIP data type of input.
  /// @param numTokens Number of token rows to reduce.
  /// @param blockNum  GPU blocks to launch (-1 = auto).
  /// @param warpPerBlock Warps per block (-1 = auto).
  /// @param stream    HIP stream.
  void Launch(void* input, hipDataType inputType, int numTokens, int blockNum = -1,
              int warpPerBlock = -1, hipStream_t stream = 0);

  /// Pointer to output buffer (in symmetric memory).
  void* OutputPtr() const { return shmemOutBuf->Get(); }

  int rank;
  int worldSize;
  int maxNumTokens;
  int hiddenDim;

  // Symmetric memory buffers
  mori::application::SymmMemObjPtr shmemInpBuf;  // P2P input staging
  mori::application::SymmMemObjPtr shmemOutBuf;   // output

  // Cross-device barrier
  mori::application::SymmMemObjPtr crossDeviceBarrierMemObj;
  uint64_t* crossDeviceBarrierFlag{nullptr};
  uint32_t* gridBarrier{nullptr};

  int multiProcessorCount{0};
};

}  // namespace allreduce
}  // namespace mori
