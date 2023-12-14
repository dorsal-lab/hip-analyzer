/** \file managed_queue_info.cpp
 * \brief GPU Queue host-side handler for dynamic memory tracing
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#include "hip_instrumentation/managed_queue_info.hpp"
#include "hip_instrumentation/hip_trace_manager.hpp"

namespace hip {

GlobalMemoryQueueInfo::GlobalMemoryQueueInfo(size_t elem_size,
                                             size_t buffer_size)
    : elem_size(elem_size) {
    // Get the timestamp for unique identification
    auto now = std::chrono::steady_clock::now();
    stamp = std::chrono::duration_cast<std::chrono::microseconds>(
                now.time_since_epoch())
                .count();

    cpu_queue.resize(buffer_size);
}

GlobalMemoryQueueInfo::GlobalMemoryTrace* GlobalMemoryQueueInfo::toDevice() {
    GlobalMemoryTrace* device_ptr;
    hip::check(hipMalloc(&device_ptr, sizeof(GlobalMemoryTrace)));

    hip::check(hipMalloc(&cpu_trace.current, cpu_queue.size()));
    cpu_trace.end =
        reinterpret_cast<std::byte*>(cpu_trace.current) + cpu_queue.size();

    hip::check(hipMemcpy(device_ptr, &cpu_trace, sizeof(GlobalMemoryTrace),
                         hipMemcpyHostToDevice));

    std::cerr << "GlobalMemoryQueueInfo : " << device_ptr << ' '
              << cpu_trace.current << '\n';

    return device_ptr;
}

void GlobalMemoryQueueInfo::fromDevice(
    GlobalMemoryQueueInfo::GlobalMemoryTrace* device_ptr) {
    GlobalMemoryTrace gpu_trace;

    hip::check(hipMemcpy(&gpu_trace, device_ptr, sizeof(GlobalMemoryTrace),
                         hipMemcpyDeviceToHost));

    auto size = reinterpret_cast<std::byte*>(gpu_trace.current) -
                reinterpret_cast<std::byte*>(cpu_trace.current);

    std::cerr << "Size : " << size << '\n';

    cpu_queue.resize(size);
    hip::check(hipMemcpy(cpu_queue.data(), cpu_trace.current, size,
                         hipMemcpyDeviceToHost));

    cpu_trace.end = gpu_trace.current;
}

void GlobalMemoryQueueInfo::record(
    GlobalMemoryQueueInfo::GlobalMemoryTrace* device_ptr) {
    hip::HipTraceManager::getInstance().registerGlobalMemoryQueue(*this,
                                                                  device_ptr);
}

} // namespace hip
