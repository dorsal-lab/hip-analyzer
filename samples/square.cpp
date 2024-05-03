#include "hip/hip_runtime.h"

#include <charconv>
#include <iostream>
#include <optional>

#include <execinfo.h>
#include <unistd.h>

inline void check(hipError_t err) {
    if (err != hipSuccess) {
        std::cerr << "error : " << hipGetErrorString(err) << " (" << err
                  << ")\nBacktrace :\n";

        void* fds[32];
        size_t size;

        size = backtrace(fds, 32);
        backtrace_symbols_fd(fds, size, STDERR_FILENO);

        throw std::runtime_error(std::string("Encountered hip error ") +
                                 hipGetErrorString(err));
    }
}

/*
 * Square each element in the array A and write to array C.
 */
template <typename T>
__global__ void vector_square(T* C_d, const T* A_d, size_t N) {
    size_t offset = (blockIdx.x * blockDim.x + threadIdx.x);
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = offset; i < N; i += stride) {
        C_d[i] = A_d[i] * A_d[i];
    }
}

std::optional<uint64_t> parse_env(const std::string& env) {
    auto buffer_size = std::getenv(env.c_str());
    if (buffer_size == nullptr) {
        return std::nullopt;
    }

    uint64_t val;
    std::string str{buffer_size};
    if (std::from_chars(str.data(), str.data() + str.length(), val).ec !=
        std::errc()) {
        std::cerr << "Could not parse u64 token : " << str << '\n';
        return std::nullopt;
    }

    return val;
}

int main(int argc, char* argv[]) {
    float *A_d, *C_d;
    float *A_h, *C_h;
    size_t N = parse_env("SQUARE_SIZE").value_or(8182);
    size_t Nbytes = N * sizeof(float);

    A_h = (float*)malloc(Nbytes);

    C_h = (float*)malloc(Nbytes);

    // Fill with Phi + i
    for (size_t i = 0; i < N; i++) {
        A_h[i] = 1.618f + i;
    }

    check(hipMalloc(&A_d, Nbytes));
    check(hipMalloc(&C_d, Nbytes));

    check(hipMemcpy(A_d, A_h, Nbytes, hipMemcpyHostToDevice));

    const unsigned blocks = parse_env("SQUARE_TOTAL_SIZE").value_or(512);
    const unsigned threadsPerBlock = parse_env("SQUARE_WG_SIZE").value_or(256);

    const auto nb_it = parse_env("SQUARE_NB_IT").value_or(1);

    for (auto i = 0u; i < nb_it; ++i) {
        vector_square<<<dim3(blocks), dim3(threadsPerBlock)>>>(C_d, A_d, N);

        check(hipDeviceSynchronize());
        check(hipMemcpy(A_d, C_d, Nbytes, hipMemcpyHostToHost));
    }

    check(hipMemcpy(C_h, C_d, Nbytes, hipMemcpyDeviceToHost));
}
