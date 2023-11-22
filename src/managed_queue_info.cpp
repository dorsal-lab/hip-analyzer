/** \file managed_queue_info.cpp
 * \brief GPU Queue host-side handler for dynamic memory tracing
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#include "hip_instrumentation/managed_queue_info.hpp"
#include "hip_instrumentation/hip_trace_manager.hpp"

namespace hip {

GlobalMemoryQueueInfo::GlobalMemoryQueueInfo(size_t elem_size,
                                             size_t buffer_size) {
    cpu_queue.resize(buffer_size);
}

GlobalMemoryQueueInfo::GlobalMemoryTrace* GlobalMemoryQueueInfo::toDevice() {
    hip::check(hipMalloc(&cpu_trace.current, cpu_queue.size()));
    cpu_trace.end =
        reinterpret_cast<std::byte*>(cpu_trace.current) + cpu_queue.size();

    GlobalMemoryTrace* device_ptr;
    hip::check(hipMalloc(&device_ptr, sizeof(GlobalMemoryTrace)));
    hip::check(hipMemcpy(device_ptr, &cpu_trace, sizeof(GlobalMemoryTrace),
                         hipMemcpyHostToDevice));

    return device_ptr;
}

void GlobalMemoryQueueInfo::fromDevice(
    GlobalMemoryQueueInfo::GlobalMemoryTrace* device_ptr) {
    GlobalMemoryTrace gpu_trace;

    hip::check(hipMemcpy(&gpu_trace, device_ptr, sizeof(GlobalMemoryTrace),
                         hipMemcpyDeviceToHost));
}

void GlobalMemoryQueueInfo::record(
    GlobalMemoryQueueInfo::GlobalMemoryTrace* device_ptr) {
    throw std::runtime_error("Unimplemented");
}

} // namespace hip
