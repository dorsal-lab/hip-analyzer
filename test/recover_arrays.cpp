/** \file recover_arrays.cpp
 * \brief StateRecoverer test case
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#include "hip_instrumentation/hip_instrumentation.hpp"
#include "hip_instrumentation/state_recoverer.hpp"

#include <iostream>

void checkValue(float* gpu_pointer, float expected_val) {
    auto value = std::make_unique<float>();
    hip::check(hipMemcpy(value.get(), gpu_pointer, sizeof(float),
                         hipMemcpyDeviceToHost));

    if (*value != expected_val) {
        std::cout << "Unexpected value, expected " << expected_val
                  << " but got " << *value << '\n';
        throw std::runtime_error("Unexpected value");
    }
}

int main() {
    hip::init();

    constexpr auto n_elements = 64u;

    std::vector<float> cpu_values(n_elements, 1.f);
    float* gpu_values;

    auto commit = [&]() {
        hip::check(hipMemcpy(gpu_values, cpu_values.data(),
                             n_elements * sizeof(float),
                             hipMemcpyHostToDevice));
    };

    hip::check(hipMalloc(&gpu_values, n_elements * sizeof(float)));

    // Initial values : everything is equal to 1.f
    commit();

    // Save state
    hip::StateRecoverer recoverer;
    recoverer.saveState({{gpu_values, n_elements}});

    // Modify something
    cpu_values[0] = 2.f;
    commit();
    recoverer.registerCallArgs(gpu_values, n_elements);

    // Assert that the copy was successful
    checkValue(gpu_values, 2.f);

    // Rollback & check
    recoverer.rollback();
    checkValue(gpu_values, 1.f);

    return 0;
}
