/** \file replace_main.c
 * \brief Replace main and execute it a few times
 *
 * \author Sébastien Darche
 */

#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

#include "hip_instrumentation/hip_trace_manager.hpp"

// Pointer to the original main
static int (*main_ptr)(int, char**, char**);

int hijacked(int argc, char** argv, char** envp) {
    int ret = 0;
    unsigned int trip_count = 1;

    const char* env = getenv("RODINIA_REPEAT");
    if (env) {
        sscanf(env, "%d", &trip_count);
    }

    for (unsigned int i = 0; i < trip_count; ++i) {
        ret |= main_ptr(argc, argv, envp);
        hip::HipTraceManager::getInstance().flush();
    }

    return ret;
}

extern "C" {
int __libc_start_main(int (*main)(int, char**, char**), int argc, char** argv,
                      int (*init)(int, char**, char**), void (*fini)(void),
                      void (*rtld_fini)(void), void* stack_end) {
    main_ptr = main;

    // Call the actual libc_start_main
    auto start_main = reinterpret_cast<typeof(&__libc_start_main)>(
        dlsym(RTLD_NEXT, "__libc_start_main"));
    return start_main(hijacked, argc, argv, init, fini, rtld_fini, stack_end);
}
}
