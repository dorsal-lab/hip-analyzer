# Hip Analyzer tests


# ----- block_db ----- #

add_executable(
    block_db
    block_db.cpp
)

target_link_libraries(block_db hip_instrumentation)
