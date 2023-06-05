# Hip Analyzer tests

project(hip-analyzer-tests LANGUAGES C CXX)

if(ENABLE_TRACEPOINTS)
    #add_definitions(-DLTTNG_UST_TRACEPOINT_DEFINE)
endif()

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
    COMMAND ${ROCM_PATH}/hip/bin/hipcc -fPIE -c -I${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/test/gpu_enqueue.cpp -std=c++20 -o ${CMAKE_CURRENT_BINARY_DIR}/gpu_enqueue.o
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
    COMMAND ${ROCM_PATH}/hip/bin/hipcc -fPIE -c -I${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/test/queue_info.cpp -std=c++20 -o ${CMAKE_CURRENT_BINARY_DIR}/queue_info.o
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
    COMMAND ${ROCM_PATH}/hip/bin/hipcc -fPIE -c -I${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/test/gpu_queue_thread.cpp -std=c++20 -o ${CMAKE_CURRENT_BINARY_DIR}/gpu_queue_thread.o
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
    COMMAND ${ROCM_PATH}/hip/bin/hipcc -fPIE -c -I${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/test/gpu_queue_wave.cpp -std=c++20 -o ${CMAKE_CURRENT_BINARY_DIR}/gpu_queue_wave.o
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
    COMMAND ${ROCM_PATH}/hip/bin/hipcc -fPIE -c -I${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/test/wavestate.cpp -std=c++20 -o ${CMAKE_CURRENT_BINARY_DIR}/wavestate.o
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
