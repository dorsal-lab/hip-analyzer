/** \file hip_instrumentation_cbindings.hpp
 * \brief Kernel instrumentation embedded code C bindings for IR manipulation
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#include <cstdint>

extern "C" {

struct hipKernelInfo;
struct hipInstrumenter;

typedef uint8_t counter_t;

hipInstrumenter* hipNewInstrumenter(hipKernelInfo*);
void hipInstrumenterLoadDb(hipInstrumenter*, const char* file_name);
counter_t* hipInstrumenterToDevice(hipInstrumenter*);
void hipInstrumenterFromDevice(hipInstrumenter*, void*);

void freeHipinstrumenter(hipInstrumenter*);
}
