/** \file state_recoverer.cpp
 * \brief GPU memory state recoverer to ensure deterministic kernel execution
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#include "hip_instrumentation/state_recoverer.hpp"
#include "hip_instrumentation/hip_utils.hpp"

#include "hip/hip_runtime.h"

namespace hip {

StateRecoverer::~StateRecoverer() {
    // Need to free all allocated values, unfortunately we have to manage the
    // memory ourselves
    for (auto [tagged_ptr, cpu_ptr] : saved_values) {
        delete[] cpu_ptr;
    }
}

void StateRecoverer::saveState(const std::vector<TaggedPointer>& pointers) {
    for (auto& ptr : pointers) {
        uint8_t* cpu_ptr = saveElement(ptr);
        saved_values.emplace(ptr, cpu_ptr);
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
        hip::check(hipMemcpy(const_cast<void*>(tagged_ptr.ptr), cpu_ptr,
                             tagged_ptr.size, hipMemcpyHostToDevice));
    }
}

} // namespace hip
