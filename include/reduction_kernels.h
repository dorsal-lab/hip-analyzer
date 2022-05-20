/** \file reduction_kernels.h
 * \brief GPU-local data analysis and reductions
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#pragma once

#include "basic_block.hpp"

namespace hip {

struct LaunchGeometry {
    uint32_t thread_count;
    uint32_t block_count;
    uint32_t bb_count;
};

struct BlockUsage {
    uint32_t count;
    uint32_t flops;
};

/** \fn reduceFlops
 * \brief Compute the number of flops per basic block. 1-dimensionnal array
 *
 * \param instr_ptr Instrumentation data pointer
 * \param geometry Launch geometry of the original kernel (thus specifying the
 * size of the instrumentation)
 * \param blocks_info Array of block data in its normalized form \ref
 * hip::BasicBlock
 * \param buffer Pre-allocated array for scratch-pad operations, of size
 * gridDim.x * blockDim.x * bb_count
 * \param output Output array of size gridDim.x * bb_count
 */

__global__ void reduceFlops(const uint8_t* instr_ptr,
                            hip::LaunchGeometry geometry,
                            const hip::BasicBlock* blocks_info,
                            hip::BlockUsage* buffer, hip::BlockUsage* output) {
    // Parallel reduction
    uint32_t tot_threads = geometry.thread_count * geometry.block_count;

    hip::BlockUsage* blocks_thread =
        &buffer[(blockIdx.x * blockDim.x + threadIdx.x) * geometry.bb_count];

    // Phase 0 : init local buffers

    for (auto bb = 0u; bb < geometry.bb_count; ++bb) {
        blocks_thread[bb] = {0u, 0u};
    }

    // Phase 1 : accumulate thread-local values

    unsigned int begin = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    unsigned int end = tot_threads;

    for (auto i = begin; i < end; i += stride) {
        if (i < tot_threads) {
            for (auto bb = 0u; bb < geometry.bb_count; ++bb) {
                blocks_thread[bb].count += static_cast<uint32_t>(
                    instr_ptr[i * geometry.bb_count + bb]);
            }
        }
    }

    // Compute flops

    for (auto bb = 0u; bb < geometry.bb_count; ++bb) {
        blocks_thread[bb].flops =
            blocks_thread[bb].count * blocks_info[bb].flops;
    }

    // Phase 2 : regroup values at block-level

    __shared__ hip::BlockUsage intermediary;

    for (auto bb = 0u; bb < geometry.bb_count; ++bb) {
        if (threadIdx.x == 0) {
            intermediary = {0u, 0u};
        }

        atomicAdd(&intermediary.count, blocks_thread[bb].count);
        atomicAdd(&intermediary.flops, blocks_thread[bb].flops);

        __syncthreads();
        if (threadIdx.x == 0) {
            blocks_thread[bb] = intermediary;
        }
    }

    // Phase 3 : save values to global memory

    if (threadIdx.x == 0) {
        for (auto bb = 0u; bb < geometry.bb_count; ++bb) {
            output[blockIdx.x * 8 + bb] = blocks_thread[bb];
        }
    }
}

} // namespace hip
