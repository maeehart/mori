# Copyright © Advanced Micro Devices, Inc. All rights reserved.
# MIT License
"""
Lightweight MORI P2P intra-node all-reduce.

Uses MORI's symmetric shared memory infrastructure for direct GPU-to-GPU
P2P reads via XGMI, avoiding RCCL / NCCL overhead.

The handle allocates only the minimum symmetric-memory buffers:
  - input staging buffer (P2P-accessible by all peers)
  - output buffer
  - cross-device barrier flags
"""
from __future__ import annotations

import torch
from mori import cpp as mori_cpp


class MoriAllReduceOp:
    """P2P intra-node all-reduce using MORI symmetric shared memory.

    Args:
        rank: This GPU's rank in the group.
        world_size: Number of GPUs in the group.
        max_num_tokens: Maximum number of tokens per inference step.
        hidden_dim: Hidden dimension (K) of the tensor to reduce.
    """

    def __init__(
        self,
        rank: int,
        world_size: int,
        max_num_tokens: int,
        hidden_dim: int,
    ):
        self._handle = mori_cpp.MoriAllReduceHandle(
            rank=rank,
            world_size=world_size,
            max_num_tokens=max_num_tokens,
            hidden_dim=hidden_dim,
        )
        self._launch_func = mori_cpp.mori_allreduce_launch

    def allreduce(
        self,
        input: torch.Tensor,
        block_num: int = -1,
        warp_per_block: int = -1,
    ) -> torch.Tensor:
        """Perform all-reduce on input[M, K] across all ranks.

        Args:
            input: [M, K] contiguous bf16/fp32 tensor.
            block_num: Override default block count (-1 = auto).
            warp_per_block: Override default warps/block (-1 = auto).

        Returns:
            Output tensor [max_num_tokens, hidden_dim] in symmetric memory.
            Caller should slice to [:M] if M < max_num_tokens.
        """
        return self._launch_func(self._handle, input, block_num, warp_per_block)
