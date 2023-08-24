/** \file hip_analyzer_tracepoint_provider.cpp
 * \brief (Automatic) tracepoint definition translation unit
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#include <string>
#ifdef ENABLE_TRACEPOINTS

#define LTTNG_UST_TRACEPOINT_CREATE_PROBES
#define LTTNG_UST_TRACEPOINT_DEFINE

namespace hip {
const char* git_hash = "@GIT_HASH@";
}

#include "hip_analyzer_tracepoints.h"
#include <chrono>

namespace {
std::string dummy = []() {
    auto timestamp =
        std::chrono::steady_clock::now().time_since_epoch().count();
    lttng_ust_tracepoint(hip_instrumentation, init, hip::git_hash, timestamp);
    return std::to_string(timestamp); // As to not optimize
}();
}

#endif
