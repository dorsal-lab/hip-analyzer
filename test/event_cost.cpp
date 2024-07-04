#include "hip/hip_runtime.h"

#include <charconv>
#include <fstream>
#include <iostream>
#include <optional>
#include <vector>

#include <execinfo.h>
#include <unistd.h>

namespace gcnasm {

inline __device__ uint64_t s_memrealtime() {
    uint64_t stamp = 0u;
#ifdef __HIPCC__
    asm volatile("s_memtime %0" : "=s"(stamp) :);
#endif
    return stamp;
}

__device__ uint32_t hip_waves_per_block() {
    auto block_size = blockDim.x;
    uint32_t waves_per_block = block_size / warpSize;
    if (block_size % warpSize != 0) {
        ++waves_per_block;
    }

    return waves_per_block;
}

__device__ uint32_t wave_id_1d() {
    auto block_id = blockIdx.x;
    auto waves_per_block = hip_waves_per_block();
    auto wave_in_block = threadIdx.x / warpSize;

    return block_id * waves_per_block + wave_in_block;
}

} // namespace gcnasm

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

__global__ void kernel1(uint64_t n_it, uint64_t* begin, uint64_t* end) {

    uint64_t t_begin = gcnasm::s_memrealtime();
    for (uint64_t i = 0ull; i < n_it; ++i) {
        asm volatile("s_nop 0xf\n");
    }
    uint64_t t_end = gcnasm::s_memrealtime();

    if (threadIdx.x % warpSize == 0) {
        auto wave_id = gcnasm::wave_id_1d();
        begin[wave_id] = t_begin;
        end[wave_id] = t_end;
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
    const auto nb_repeat = parse_env("NB_REPEAT").value_or(1);
    const auto nb_it = parse_env("NB_IT").value_or(1);
    const auto nb_waves = parse_env("NB_WAVES").value_or(1);

    const char* out_filename = std::getenv("OUTPUT");
    if (out_filename == nullptr) {
        out_filename = "out.csv";
    }
    std::ofstream out(out_filename);
    out << "iteration,wave,begin,end\n";

    size_t size_alloc = nb_waves * sizeof(uint64_t);

    uint64_t *begin_device, *end_device;
    check(hipMalloc(&begin_device, size_alloc));
    check(hipMalloc(&end_device, size_alloc));

    std::vector<uint64_t> begin, end;
    begin.resize(nb_waves);
    end.resize(nb_waves);

    for (auto i = 0u; i < nb_repeat; ++i) {
        kernel1<<<nb_waves, 64>>>(nb_it, begin_device, end_device);

        check(hipDeviceSynchronize());
        check(hipMemcpy(begin.data(), begin_device, size_alloc,
                        hipMemcpyDeviceToHost));
        check(hipMemcpy(end.data(), end_device, size_alloc,
                        hipMemcpyDeviceToHost));

        for (auto wave = 0u; wave < nb_waves; ++wave) {
            out << i << ',' << wave << ',' << begin[wave] << ',' << end[wave]
                << '\n';
        }
    }

    out.close();
}
