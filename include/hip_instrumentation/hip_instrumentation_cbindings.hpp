/** \file hip_instrumentation_cbindings.hpp
 * \brief Kernel instrumentation embedded code C bindings for IR manipulation
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#include "managed_queue_info.hpp"
#include <cstdint>

namespace hip {
class CounterInstrumenter;
class StateRecoverer;
struct QueueInfo;
} // namespace hip

extern "C" {

// ----- Instrumentation ----- //

enum class CounterType : uint32_t { Thread = 0u, Wave = 1u };

uint64_t rocmStamp();

/** \fn hipNewInstrumenter
 * \brief Create a new instrumenter from the (mangled) kernel name
 */
hip::CounterInstrumenter* hipNewInstrumenter(const char* kernel_name,
                                             CounterType type);

/** \fn hipGetNextInstrumenter
 * \brief Create a new instrumenter by loading it from a tracefile
 */
hip::CounterInstrumenter* hipGetNextInstrumenter();

/** \fn hipInstrumenterToDevice
 * \brief Create the instrumentation counters
 */
void* hipInstrumenterToDevice(hip::CounterInstrumenter*);

/** \fn hipInstrumenterFromDevice
 * \brief Fetches the counters from the device
 */
void hipInstrumenterFromDevice(hip::CounterInstrumenter*, void*);

void hipInstrumenterRecord(hip::CounterInstrumenter*);

/** \fn freeHipInstrumenter
 */
void freeHipInstrumenter(hip::CounterInstrumenter*);

// ----- State recoverer ----- //

hip::StateRecoverer* hipNewStateRecoverer();

/** \fn hipMemoryManagerRegisterPointer
 * \brief Equivalent of hip::HipMemoryManager::registerCallArgs(T...),
 * register pointers as used in the shadow memory
 */
void* hipStateRecovererRegisterPointer(hip::StateRecoverer*,
                                       void* potential_ptr);

/** \fn hipMemoryManagerRollback
 * \brief Equivalent of hip::HipMemoryManager::rollback(), revert to the
 * original value of all arrays
 */
void hipStateRecovererRollback(hip::StateRecoverer*, hip::CounterInstrumenter*);

/** \fn freeHipStateRecoverer
 */
void freeHipStateRecoverer(hip::StateRecoverer*);

// ----- Event queue ----- //

enum class EventType : uint32_t {
    Event = 0,
    TaggedEvent = 1,
    WaveState = 2,
};

enum class QueueType : uint32_t {
    Thread = 0,
    Wave = 1,
};

hip::QueueInfo* newHipQueueInfo(hip::CounterInstrumenter*, EventType,
                                QueueType);

void freeHipQueueInfo(hip::QueueInfo*);

void* hipQueueInfoAllocBuffer(hip::QueueInfo*);

void* hipQueueInfoAllocOffsets(hip::QueueInfo*);

void hipQueueInfoRecord(hip::QueueInfo*, void*, void*);

hip::GlobalMemoryQueueInfo* newGlobalMemQueueInfo(size_t event_size);

hip::GlobalMemoryQueueInfo::GlobalMemoryTrace*
hipGlobalMemQueueInfoToDevice(hip::GlobalMemoryQueueInfo*);

void hipGlobalMemQueueInfoRecord(
    hip::GlobalMemoryQueueInfo*,
    hip::GlobalMemoryQueueInfo::GlobalMemoryTrace*);

void freeHipGlobalMemoryQueueInfo(hip::GlobalMemoryQueueInfo*);

hip::ChunkAllocator* newHipChunkAllocator(const char* kernel_name,
                                          size_t buffer_count,
                                          size_t buffer_size);

hip::ChunkAllocator::Registry* hipChunkAllocatorToDevice(hip::ChunkAllocator*);

void hipChunkAllocatorRecord(hip::ChunkAllocator*, uint64_t stamp);

void freeChunkAllocator(hip::ChunkAllocator*);
}
