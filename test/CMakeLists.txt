# Hip Analyzer tests

project(hip-analyzer-tests LANGUAGES C CXX)

# ----- block_db ----- #

add_executable(
    block_db
    block_db.cpp
)

target_link_libraries(block_db hip_instrumentation)


# ----- load_bin ----- #

add_executable(
    load_bin
    load_bin.cpp
)

target_link_libraries(load_bin hip_instrumentation LLVMSupport)

# ----- load_csv ----- #

add_executable(
    load_csv
    load_csv.cpp
)

target_link_libraries(load_csv hip_instrumentation LLVMSupport)

# ----- arithmetic_intensity ----- #

add_executable(
    arithmetic_intensity
    arithmetic_intensity.cpp
)

target_link_libraries(arithmetic_intensity hip_instrumentation LLVMSupport)

# ----- gpu_benchmark ----- #

add_executable(
    gpu_benchmark
    gpu_benchmark.cpp
)

target_link_libraries(gpu_benchmark hip_instrumentation LLVMSupport)

# ----- recover_arrays ----- #

add_executable(
    recover_arrays
    recover_arrays.cpp
)

target_link_libraries(recover_arrays hip_instrumentation)

# ----- gpu_queue ----- #

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/gpu_enqueue.o
    COMMAND ${ROCM_PATH}/bin/hipcc -fPIE -c -I${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/test/gpu_enqueue.cpp -std=c++20 -o ${CMAKE_CURRENT_BINARY_DIR}/gpu_enqueue.o
    DEPENDS 
        ${CMAKE_SOURCE_DIR}/include/hip_instrumentation/hip_instrumentation.hpp
        ${CMAKE_SOURCE_DIR}/include/hip_instrumentation/hip_instrumentation_cbindings.hpp
        ${CMAKE_SOURCE_DIR}/include/hip_instrumentation/hip_utils.hpp
        ${CMAKE_SOURCE_DIR}/include/hip_instrumentation/gpu_queue.hpp
        ${CMAKE_SOURCE_DIR}/test/gpu_enqueue.cpp
    VERBATIM
)

add_executable(gpu_enqueue ${CMAKE_CURRENT_BINARY_DIR}/gpu_enqueue.o)
target_link_libraries(gpu_enqueue hip_instrumentation)
set_target_properties(gpu_enqueue PROPERTIES LINKER_LANGUAGE CXX)

# ----- queue_info ----- #

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/queue_info.o
    COMMAND ${ROCM_PATH}/bin/hipcc -fPIE -c -I${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/test/queue_info.cpp -std=c++20 -o ${CMAKE_CURRENT_BINARY_DIR}/queue_info.o
    DEPENDS 
        ${CMAKE_SOURCE_DIR}/include/hip_instrumentation/hip_instrumentation.hpp
        ${CMAKE_SOURCE_DIR}/include/hip_instrumentation/hip_instrumentation_cbindings.hpp
        ${CMAKE_SOURCE_DIR}/include/hip_instrumentation/hip_utils.hpp
        ${CMAKE_SOURCE_DIR}/include/hip_instrumentation/gpu_queue.hpp
        ${CMAKE_SOURCE_DIR}/include/hip_instrumentation/queue_info.hpp
        ${CMAKE_SOURCE_DIR}/test/queue_info.cpp
    VERBATIM
)

add_executable(queue_info ${CMAKE_CURRENT_BINARY_DIR}/queue_info.o)
target_link_libraries(queue_info hip_instrumentation)
set_target_properties(queue_info PROPERTIES LINKER_LANGUAGE CXX)


# ----- gpu_queue_thread ----- #

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/gpu_queue_thread.o
    COMMAND ${ROCM_PATH}/bin/hipcc -fPIE -c -I${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/test/gpu_queue_thread.cpp -std=c++20 -o ${CMAKE_CURRENT_BINARY_DIR}/gpu_queue_thread.o
    DEPENDS 
        ${CMAKE_SOURCE_DIR}/include/hip_instrumentation/hip_instrumentation.hpp
        ${CMAKE_SOURCE_DIR}/include/hip_instrumentation/hip_instrumentation_cbindings.hpp
        ${CMAKE_SOURCE_DIR}/include/hip_instrumentation/hip_utils.hpp
        ${CMAKE_SOURCE_DIR}/include/hip_instrumentation/gpu_queue.hpp
        ${CMAKE_SOURCE_DIR}/include/hip_instrumentation/queue_info.hpp
        ${CMAKE_SOURCE_DIR}/test/gpu_queue_thread.cpp
    VERBATIM
)

add_executable(gpu_queue_thread ${CMAKE_CURRENT_BINARY_DIR}/gpu_queue_thread.o)
target_link_libraries(gpu_queue_thread hip_instrumentation)
set_target_properties(gpu_queue_thread PROPERTIES LINKER_LANGUAGE CXX)

# ----- gpu_queue_wave ----- #

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/gpu_queue_wave.o
    COMMAND ${ROCM_PATH}/bin/hipcc -fPIE -c -I${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/test/gpu_queue_wave.cpp -std=c++20 -o ${CMAKE_CURRENT_BINARY_DIR}/gpu_queue_wave.o
    DEPENDS 
        ${CMAKE_SOURCE_DIR}/include/hip_instrumentation/hip_instrumentation.hpp
        ${CMAKE_SOURCE_DIR}/include/hip_instrumentation/hip_instrumentation_cbindings.hpp
        ${CMAKE_SOURCE_DIR}/include/hip_instrumentation/hip_utils.hpp
        ${CMAKE_SOURCE_DIR}/include/hip_instrumentation/gpu_queue.hpp
        ${CMAKE_SOURCE_DIR}/include/hip_instrumentation/queue_info.hpp
        ${CMAKE_SOURCE_DIR}/test/gpu_queue_wave.cpp
    VERBATIM
)

add_executable(gpu_queue_wave ${CMAKE_CURRENT_BINARY_DIR}/gpu_queue_wave.o)
target_link_libraries(gpu_queue_wave hip_instrumentation)
set_target_properties(gpu_queue_wave PROPERTIES LINKER_LANGUAGE CXX)

# ----- wavestate ----- #

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/wavestate.o
    COMMAND ${ROCM_PATH}/bin/hipcc -fPIE -c -I${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/test/wavestate.cpp -std=c++20 -o ${CMAKE_CURRENT_BINARY_DIR}/wavestate.o
    DEPENDS
        ${CMAKE_SOURCE_DIR}/src/pass/gpu_pass_instr.hip
        ${CMAKE_SOURCE_DIR}/include/hip_instrumentation/hip_utils.hpp
        ${CMAKE_SOURCE_DIR}/include/hip_instrumentation/gpu_queue.hpp
        ${CMAKE_SOURCE_DIR}/include/hip_instrumentation/queue_info.hpp
        ${CMAKE_SOURCE_DIR}/include/hip_instrumentation/hip_instrumentation_cbindings.hpp
        ${CMAKE_SOURCE_DIR}/test/wavestate.cpp
    VERBATIM
)

add_executable(wavestate ${CMAKE_CURRENT_BINARY_DIR}/wavestate.o)
target_link_libraries(wavestate hip_instrumentation)
set_target_properties(wavestate PROPERTIES LINKER_LANGUAGE CXX)

# ----- wavestate_asm ----- #

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/wavestate_asm.o
    COMMAND ${ROCM_PATH}/bin/hipcc -fPIE -c -I${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/test/wavestate_asm.cpp -std=c++20 -o ${CMAKE_CURRENT_BINARY_DIR}/wavestate_asm.o
    DEPENDS
        ${CMAKE_SOURCE_DIR}/src/pass/gpu_pass_instr.hip
        ${CMAKE_SOURCE_DIR}/include/hip_instrumentation/hip_utils.hpp
        ${CMAKE_SOURCE_DIR}/include/hip_instrumentation/gpu_queue.hpp
        ${CMAKE_SOURCE_DIR}/include/hip_instrumentation/queue_info.hpp
        ${CMAKE_SOURCE_DIR}/include/hip_instrumentation/hip_instrumentation_cbindings.hpp
        ${CMAKE_SOURCE_DIR}/test/wavestate_asm.cpp
    VERBATIM
)

add_executable(wavestate_asm ${CMAKE_CURRENT_BINARY_DIR}/wavestate_asm.o)
target_link_libraries(wavestate_asm hip_instrumentation)
set_target_properties(wavestate_asm PROPERTIES LINKER_LANGUAGE CXX)

# ----- get_wave_trace_offset ----- #

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/get_wave_trace_offset.o
    COMMAND ${ROCM_PATH}/bin/hipcc -fPIE -c -I${CMAKE_SOURCE_DIR}/include -O3 ${CMAKE_SOURCE_DIR}/test/get_wave_trace_offset.cpp -std=c++20 -o ${CMAKE_CURRENT_BINARY_DIR}/get_wave_trace_offset.o
    DEPENDS
        ${CMAKE_SOURCE_DIR}/src/pass/gpu_pass_instr.hip
        ${CMAKE_SOURCE_DIR}/include/hip_instrumentation/hip_utils.hpp
        ${CMAKE_SOURCE_DIR}/include/hip_instrumentation/gpu_queue.hpp
        ${CMAKE_SOURCE_DIR}/include/hip_instrumentation/queue_info.hpp
        ${CMAKE_SOURCE_DIR}/include/hip_instrumentation/hip_instrumentation_cbindings.hpp
        ${CMAKE_SOURCE_DIR}/test/get_wave_trace_offset.cpp
    VERBATIM
)

add_executable(get_wave_trace_offset ${CMAKE_CURRENT_BINARY_DIR}/get_wave_trace_offset.o)
target_link_libraries(get_wave_trace_offset hip_instrumentation)
set_target_properties(get_wave_trace_offset PROPERTIES LINKER_LANGUAGE CXX)

# ----- replace_main.so ----- #

add_library(replace_main SHARED
    ${CMAKE_SOURCE_DIR}/test/replace_main.c
)
target_compile_options(replace_main PRIVATE "-fPIC")
target_link_libraries(replace_main dl)

# ----- read_trace ----- #

add_executable(read_trace
    read_trace.cpp
)

target_link_libraries(read_trace hip_instrumentation LLVMSupport)

# ----- basic_blocks_freq ----- #

add_executable(basic_blocks_freq
    basic_blocks_freq.cpp
)

target_link_libraries(basic_blocks_freq hip_instrumentation LLVMSupport)

# -----global_atomics_tracing ----- #

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/global_atomics_tracing.o
    COMMAND ${ROCM_PATH}/bin/hipcc -fPIE -c -I${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/test/global_atomics_tracing.cpp -std=c++20 -o ${CMAKE_CURRENT_BINARY_DIR}/global_atomics_tracing.o
    DEPENDS
        ${CMAKE_SOURCE_DIR}/include/hip_instrumentation/hip_utils.hpp
        ${CMAKE_SOURCE_DIR}/include/hip_instrumentation/gpu_queue.hpp
        ${CMAKE_SOURCE_DIR}/include/hip_instrumentation/managed_queue_info.hpp
        ${CMAKE_SOURCE_DIR}/include/hip_instrumentation/hip_instrumentation_cbindings.hpp
        ${CMAKE_SOURCE_DIR}/test/global_atomics_tracing.cpp
    VERBATIM
)

add_executable(global_atomics_tracing ${CMAKE_CURRENT_BINARY_DIR}/global_atomics_tracing.o)
target_link_libraries(global_atomics_tracing hip_instrumentation)
set_target_properties(global_atomics_tracing PROPERTIES LINKER_LANGUAGE CXX)

# -----global_atomics_tracing_asm ----- #

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/global_atomics_tracing_asm.o
    COMMAND ${ROCM_PATH}/bin/hipcc -fPIE -c -I${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/test/global_atomics_tracing_asm.cpp -std=c++20 -o ${CMAKE_CURRENT_BINARY_DIR}/global_atomics_tracing_asm.o
    DEPENDS
        ${CMAKE_SOURCE_DIR}/include/hip_instrumentation/hip_utils.hpp
        ${CMAKE_SOURCE_DIR}/include/hip_instrumentation/gpu_queue.hpp
        ${CMAKE_SOURCE_DIR}/include/hip_instrumentation/managed_queue_info.hpp
        ${CMAKE_SOURCE_DIR}/include/hip_instrumentation/hip_instrumentation_cbindings.hpp
        ${CMAKE_SOURCE_DIR}/test/global_atomics_tracing_asm.cpp
    VERBATIM
)

add_executable(global_atomics_tracing_asm ${CMAKE_CURRENT_BINARY_DIR}/global_atomics_tracing_asm.o)
target_link_libraries(global_atomics_tracing_asm hip_instrumentation)
set_target_properties(global_atomics_tracing_asm PROPERTIES LINKER_LANGUAGE CXX)

# ----- chunk_allocator ----- #

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/chunk_allocator.o
    COMMAND ${ROCM_PATH}/bin/hipcc -fPIE -c -I${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/test/chunk_allocator.cpp -std=c++20 -o ${CMAKE_CURRENT_BINARY_DIR}/chunk_allocator.o
    DEPENDS
        ${CMAKE_SOURCE_DIR}/include/hip_instrumentation/hip_utils.hpp
        ${CMAKE_SOURCE_DIR}/include/hip_instrumentation/gpu_queue.hpp
        ${CMAKE_SOURCE_DIR}/include/hip_instrumentation/managed_queue_info.hpp
        ${CMAKE_SOURCE_DIR}/include/hip_instrumentation/hip_instrumentation_cbindings.hpp
        ${CMAKE_SOURCE_DIR}/test/chunk_allocator.cpp
    VERBATIM
)

add_executable(chunk_allocator ${CMAKE_CURRENT_BINARY_DIR}/chunk_allocator.o)
target_link_libraries(chunk_allocator hip_instrumentation)
set_target_properties(chunk_allocator PROPERTIES LINKER_LANGUAGE CXX)



# ----- read_trace ----- #

add_executable(count_allocations
    count_allocations.cpp
)

target_link_libraries(count_allocations hip_instrumentation LLVMSupport)
