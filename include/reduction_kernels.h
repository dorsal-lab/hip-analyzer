/** \file reduction_kernels.h
 * \brief GPU-local data analysis and reductions
 *
 * \author Sébastien Darche <polymtl.ca>
 */

#include "basic_block.hpp"

namespace hip {

struct LaunchGeometry {
    uint32_t thread_count;
    uint32_t block_count;
    uint32_t bb_count;
};

struct BlockUsage {
    uint32_t count = 0u;
    uint32_t flops = 0u;
} __attribute__((packed));

} // namespace hip

/** \fn reduceFlops
 * \brief Compute the number of flops per basic block. 1-dimensionnal array
 *
 * \param instr_ptr Instrumentation data pointer
 * \param geometry Launch geometry of the original kernel (thus specifying the
 * size of the instrumentation)
 * \param blocks_info Array of block data in its normalized form \ref
 * hip::BasicBlock
 * \param output Output array of size gridDim.x * bb_count
 *
 * \returns output :
 */

__global__ void reduceFlops(const uint8_t* instr_ptr,
                            hip::LaunchGeometry geometry,
                            const hip::BasicBlock* blocks_info,
                            BlockUsage* output) {
    // Parallel reduction
    uint32_t tot_threads = geometry.thread_count * geometry.block_count;
    BlockUsage blocks_thread[geometry.block_count];

    // Phase 1 : accumulate thread-local values

    unsigned int begin = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    unsigned int end = tot_threads;

    for (auto i = begin; i < end; i += stride) {
        if (i < tot_threads) {
            for (auto bb = 0u; bb < geometry.bb_count; ++bb) {
                blocks_thread[bb].count += instr_ptr[i * 8 + bb];
            }
        }
    }

    // Compute flops

    for (auto bb = 0u; bb < geometry.bb_count; ++bb) {
        blocks_thread[bb].flops =
            blocks_thread[bb].count * blocks_info[bb].flops;
    }

    // Phase 2 : regroup values at block-level

    __shared__ BlockUsage intermediary[geometry.bb_count];

    for (auto bb = 0u; bb < geometry.bb_count; ++bb) {
        atomicAdd(intermediary[bb].count, blocks_thread[bb].count);
        atomicAdd(intermediary[bb].flops, blocks_thread[bb].flops);
    }

    // Phase 3 : save values to global memory

    if (threadIdx.x == 0) {
        for (auto bb = 0u; bb < geometry.bb_count; ++bb) {
            output[blockIdx.x * 8 + bb] = intermediary[bb];
        }
    }
}
