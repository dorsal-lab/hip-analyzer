/** \file hip_instrumentation_cbindings.hpp
 * \brief Kernel instrumentation embedded code C bindings for IR manipulation
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#include <cstdint>

extern "C" {

struct hipInstrumenter;
struct hipStateRecoverer;
struct hipQueueInfo;

typedef uint8_t counter_t;

// ----- Instrumentation ----- //

/** \fn hipNewInstrumenter
 * \brief Create a new instrumenter from the (mangled) kernel name
 */
hipInstrumenter* hipNewInstrumenter(const char* kernel_name);

/** \fn hipInstrumenterToDevice
 * \brief Create the instrumentation counters
 */
counter_t* hipInstrumenterToDevice(hipInstrumenter*);

/** \fn hipInstrumenterFromDevice
 * \brief Fetches the counters from the device
 */
void hipInstrumenterFromDevice(hipInstrumenter*, void*);

void hipInstrumenterRecord(hipInstrumenter*);

/** \fn freeHipInstrumenter
 */
void freeHipInstrumenter(hipInstrumenter*);

// ----- State recoverer ----- //

hipStateRecoverer* hipNewStateRecoverer();

/** \fn hipMemoryManagerRegisterPointer
 * \brief Equivalent of hip::HipMemoryManager::registerCallArgs(T...), register
 * pointers as used in the shadow memory
 */
void hipStateRecovererRegisterPointer(hipStateRecoverer*, void* potential_ptr);

/** \fn hipMemoryManagerRollback
 * \brief Equivalent of hip::HipMemoryManager::rollback(), revert to the
 * original value of all arrays
 */
void hipStateRecovererRollback(hipStateRecoverer*);

/** \fn freeHipStateRecoverer
 *
 */
void freeHipStateRecoverer(hipStateRecoverer*);

// ----- Event queue ----- //

enum class EventType {
    Event = 0,
    TaggedEvent = 1,
    WaveState = 2,
};

enum class QueueType {
    Thread = 0,
    Wave = 1,
};

hipQueueInfo* newHipQueueInfo(hipInstrumenter*, EventType, QueueType);

void* hipQueueInfoAllocBuffer(hipQueueInfo*);

void* hipQueueInfoAllocOffsets(hipQueueInfo*);

void hipQueueInfoRecord(hipQueueInfo*);

void hipQueueInfoFromDevice(hipQueueInfo*, void*);
}
