/** \file state_recoverer.cpp
 * \brief GPU memory state recoverer to ensure deterministic kernel execution
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#include "hip_instrumentation/state_recoverer.hpp"
#include "hip_instrumentation/hip_utils.hpp"

#include "hip/hip_runtime.h"

#include <dlfcn.h>

namespace hip {

StateRecoverer::~StateRecoverer() {
    // Need to free all allocated values, unfortunately we have to manage the
    // memory ourselves
    for (auto& [tagged_ptr, cpu_ptr] : saved_values) {
        delete[] cpu_ptr;
    }
}

void StateRecoverer::saveState(const std::vector<TaggedPointer>& pointers) {
    for (auto& ptr : pointers) {
        uint8_t* cpu_ptr = saveElement(ptr);
        saved_values.emplace_back(ptr, cpu_ptr);
    }
}

uint8_t* StateRecoverer::saveElement(const TaggedPointer& ptr) {
    // Allocate a byte array to store the values

    uint8_t* cpu_ptr = new uint8_t[ptr.size];

    // Copy the data back from the GPU

    hip::check(hipMemcpy(cpu_ptr, ptr.ptr, ptr.size, hipMemcpyDeviceToHost));

    return cpu_ptr;
}

void StateRecoverer::rollback() const {
    for (auto [tagged_ptr, cpu_ptr] : saved_values) {
        if (tagged_ptr.dirty) {
            hip::check(hipMemcpy(const_cast<void*>(tagged_ptr.ptr), cpu_ptr,
                                 tagged_ptr.size, hipMemcpyHostToDevice));

            tagged_ptr.dirty = false;
        }
    }
}

std::unique_ptr<HipMemoryManager> HipMemoryManager::instance;

HipMemoryManager::HipMemoryManager() {
    so_handle = reinterpret_cast<char*>(dlopen("libamdhip64.so", RTLD_LAZY));
    if (!so_handle) {
        throw std::runtime_error("HipMemoryManager::HipMemoryManager() : Could "
                                 "not load shared object libamdhip64.so");
    }

    hipMallocHandler =
        reinterpret_cast<hipError_t (*)(void** ptr, size_t size)>(
            dlsym(so_handle, "hipMalloc"));
    if (!hipMallocHandler) {
        throw std::runtime_error("HipMemoryManager::HipMemoryManager() : Could "
                                 "not load hipMalloc symbol");
    }

    hipFreeHandler = reinterpret_cast<hipError_t (*)(void* ptr)>(
        dlsym(so_handle, "hipFree"));
    if (!hipFreeHandler) {
        throw std::runtime_error("HipMemoryManager::HipMemoryManager() : Could "
                                 "not load hipFree symbol");
    }
}

hipError_t HipMemoryManager::hipMallocWrapper(void** ptr, size_t size,
                                              size_t el_size) {
    hipError_t err = hipMallocHandler(ptr, size);
    if (err == hipSuccess) {
        TaggedPointer tagged_ptr{static_cast<uint8_t*>(*ptr), size, el_size};

        alloc_map.emplace(std::make_pair(*ptr, tagged_ptr));
    }
    return err;
}

hipError_t HipMemoryManager::hipFreeWrapper(void* ptr) {
    hipError_t err = hipFreeHandler(ptr);
    if (err == hipSuccess) {
        if (alloc_map.erase(ptr) == 0) {
            // Not found in map, do we need to tell the user ?
        }
    }
    return err;
}

HipMemoryManager::~HipMemoryManager() {
    for (auto& [ptr, tagged_ptr] : alloc_map) {
        std::cout << "HipMemoryManager::~HipMemoryManager : unfreed object "
                  << ptr << '\n';
    }
}

} // namespace hip

extern "C" {
hipError_t hipMalloc(void** ptr, size_t size) {
    return hip::HipMemoryManager::getInstance().hipMalloc(ptr, size);
}

hipError_t hipFree(void* ptr) {
    return hip::HipMemoryManager::getInstance().hipFree(ptr);
}
}
