/** \file gpu_functions.cpp
 * \brief Functions with device code (isolating slow compilations)
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#include "hip_instrumentation.hpp"
#include "hip_utils.hpp"
#include "reduction_kernels.h"

// TODO : Compile this, then link it properly

unsigned int hip::Instrumenter::reduceFlops(const counter_t* device_ptr,
                                            hipStream_t stream) const {

    if (blocks.empty()) {
        // The block database has to be loaded prior to reduction!
        throw std::runtime_error("BasicBlock::normalized : Empty vector");
    }

    unsigned int num_blocks = 128u; // Ultimately, how many last reductions will
                                    // we have to do on the cpu
    unsigned int threads_per_block = 128u;

    // ----- Malloc & memcopy ----- //

    // Temporary buffer
    auto buf_size = num_blocks * threads_per_block * kernel_info.basic_blocks *
                    sizeof(hip::BlockUsage);

    hip::BlockUsage* buffer_ptr;
    hip::check(hipMalloc(&buffer_ptr, buf_size));

    // Output buffer

    std::vector<hip::BlockUsage> output(num_blocks * kernel_info.basic_blocks,
                                        {0u, 0u});

    auto output_size = output.size() * sizeof(hip::BlockUsage);

    hip::BlockUsage* output_ptr;
    hip::check(hipMalloc(&output_ptr, buf_size));

    // Launch geometry

    hip::LaunchGeometry geometry{kernel_info.total_threads_per_blocks,
                                 kernel_info.total_blocks,
                                 kernel_info.basic_blocks};

    // Basic blocks
    auto blocks_info = hip::BasicBlock::normalized(blocks);
    auto blocks_info_size = blocks_info.size() * sizeof(hip::BasicBlock);

    hip::BasicBlock* blocks_info_ptr;
    hip::check(hipMalloc(&blocks_info_ptr, blocks_info_size));

    hip::check(hipMemcpy(blocks_info_ptr, blocks_info.data(), blocks_info_size,
                         hipMemcpyHostToDevice));

    // ----- Synchronization ----- //
    if (!stream) {
        hip::check(hipDeviceSynchronize());
    }

    // ----- Launch kernel ----- //

    ::hip::
        reduceFlops<<<dim3(num_blocks), dim3(threads_per_block), 0, stream>>>(
            device_ptr, geometry, blocks_info_ptr, buffer_ptr, output_ptr);

    // ----- Fetch back data ----- //

    if (!stream) {
        hip::check(hipDeviceSynchronize());
    } else {
        hip::check(hipStreamSynchronize(stream));
    }

    hip::check(hipMemcpy(output.data(), output_ptr,
                         output.size() * sizeof(hip::BlockUsage),
                         hipMemcpyDeviceToHost));

    // ----- Free device memory ----- //

    hip::check(hipFree(output_ptr));
    hip::check(hipFree(buffer_ptr));
    hip::check(hipFree(blocks_info_ptr));

    // ----- Final reduction ----- //

    unsigned int flops = 0u;
    for (auto i = 0u; i < num_blocks; ++i) {
        for (auto bb = 0u; bb < kernel_info.basic_blocks; ++bb) {
            auto block_output = output[i * kernel_info.basic_blocks + bb];

            // std::cout << i << ' ' << bb << " : " << block_output.count <<
            // '\n';

            flops += block_output.flops;
        }
    }

    return flops;
}
