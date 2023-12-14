/** \file hip_analzer_tracepoints.h
 * \brief LTTng-ust tracepoints for hip analyzer runtime
 */

#ifdef ENABLE_TRACEPOINTS

#undef LTTNG_UST_TRACEPOINT_PROVIDER
#define LTTNG_UST_TRACEPOINT_PROVIDER hip_instrumentation

#undef LTTNG_UST_TRACEPOINT_INCLUDE
#define LTTNG_UST_TRACEPOINT_INCLUDE "hip_analyzer_tracepoints.h"

#if !defined(_HIP_INSTRUMENTATION_TP_H) ||                                     \
    defined(LTTNG_UST_TRACEPOINT_HEADER_MULTI_READ)
#define _HIP_INSTRUMENTATION_TP_H

#include <lttng/tracepoint.h>

// ----- Memory operations ----- //

LTTNG_UST_TRACEPOINT_EVENT(
    hip_instrumentation, hipMalloc,
    LTTNG_UST_TP_ARGS(const void*, device_ptr, size_t, size),
    LTTNG_UST_TP_FIELDS(lttng_ust_field_integer_hex(
        uintptr_t, device_ptr, reinterpret_cast<uintptr_t>(device_ptr))
                            lttng_ust_field_integer(size_t, size, size)))

LTTNG_UST_TRACEPOINT_EVENT(hip_instrumentation, hipFree,
                           LTTNG_UST_TP_ARGS(const void*, device_ptr),
                           LTTNG_UST_TP_FIELDS(lttng_ust_field_integer_hex(
                               uintptr_t, device_ptr,
                               reinterpret_cast<uintptr_t>(device_ptr))))

// ----- HIP Instrumentation handlers ----- //

LTTNG_UST_TRACEPOINT_EVENT_CLASS(
    hip_instrumentation, instr_activity, LTTNG_UST_TP_ARGS(const void*, instr),
    LTTNG_UST_TP_FIELDS(lttng_ust_field_integer_hex(
        uintptr_t, inst, reinterpret_cast<uintptr_t>(instr))))

LTTNG_UST_TRACEPOINT_EVENT_CLASS(
    hip_instrumentation, instr_activity_ptr,
    LTTNG_UST_TP_ARGS(const void*, instr, const void*, ptr),
    LTTNG_UST_TP_FIELDS(
        lttng_ust_field_integer_hex(uintptr_t, instr,
                                    reinterpret_cast<uintptr_t>(instr))
            lttng_ust_field_integer_hex(uintptr_t, ptr,
                                        reinterpret_cast<uintptr_t>(ptr))))

LTTNG_UST_TRACEPOINT_EVENT(
    hip_instrumentation, init,
    LTTNG_UST_TP_ARGS(const char*, commit_hash, uint64_t, timestamp),
    LTTNG_UST_TP_FIELDS(lttng_ust_field_string(commit_hash, commit_hash)
                            lttng_ust_field_integer(uint64_t, timestamp,
                                                    timestamp)))
LTTNG_UST_TRACEPOINT_EVENT(
    hip_instrumentation, new_instrumenter,
    LTTNG_UST_TP_ARGS(const void*, instr, const char*, kernel),
    LTTNG_UST_TP_FIELDS(lttng_ust_field_integer_hex(
        uintptr_t, instr, reinterpret_cast<uintptr_t>(instr))
                            lttng_ust_field_string(kernel, kernel)))

LTTNG_UST_TRACEPOINT_EVENT(
    hip_instrumentation, new_replayer,
    LTTNG_UST_TP_ARGS(const void*, instr, const char*, kernel),
    LTTNG_UST_TP_FIELDS(lttng_ust_field_integer_hex(
        uintptr_t, instr, reinterpret_cast<uintptr_t>(instr))
                            lttng_ust_field_string(kernel, kernel)))

LTTNG_UST_TRACEPOINT_EVENT_INSTANCE(hip_instrumentation, instr_activity,
                                    hip_instrumentation, delete_instrumenter,
                                    LTTNG_UST_TP_ARGS(const void*, instr))

LTTNG_UST_TRACEPOINT_EVENT_INSTANCE(hip_instrumentation, instr_activity,
                                    hip_instrumentation, instr_to_device_begin,
                                    LTTNG_UST_TP_ARGS(const void*, instr))

LTTNG_UST_TRACEPOINT_EVENT_INSTANCE(hip_instrumentation, instr_activity_ptr,
                                    hip_instrumentation, instr_to_device_end,
                                    LTTNG_UST_TP_ARGS(const void*, instr,
                                                      const void*, ptr))

LTTNG_UST_TRACEPOINT_EVENT_INSTANCE(hip_instrumentation, instr_activity,
                                    hip_instrumentation, instr_sync_begin,
                                    LTTNG_UST_TP_ARGS(const void*, instr))

LTTNG_UST_TRACEPOINT_EVENT_INSTANCE(hip_instrumentation, instr_activity,
                                    hip_instrumentation, instr_sync_end,
                                    LTTNG_UST_TP_ARGS(const void*, instr))

LTTNG_UST_TRACEPOINT_EVENT_INSTANCE(hip_instrumentation, instr_activity,
                                    hip_instrumentation, instr_record_begin,
                                    LTTNG_UST_TP_ARGS(const void*, instr))

LTTNG_UST_TRACEPOINT_EVENT_INSTANCE(hip_instrumentation, instr_activity,
                                    hip_instrumentation, instr_record_end,
                                    LTTNG_UST_TP_ARGS(const void*, instr))

LTTNG_UST_TRACEPOINT_EVENT_INSTANCE(hip_instrumentation, instr_activity_ptr,
                                    hip_instrumentation,
                                    instr_from_device_begin,
                                    LTTNG_UST_TP_ARGS(const void*, instr,
                                                      const void*, ptr))

LTTNG_UST_TRACEPOINT_EVENT_INSTANCE(hip_instrumentation, instr_activity_ptr,
                                    hip_instrumentation, instr_from_device_end,
                                    LTTNG_UST_TP_ARGS(const void*, instr,
                                                      const void*, ptr))

// Queue info

LTTNG_UST_TRACEPOINT_EVENT_INSTANCE(hip_instrumentation, instr_activity_ptr,
                                    hip_instrumentation, queue_compute_begin,
                                    LTTNG_UST_TP_ARGS(const void*, instr,
                                                      const void*, ptr))

LTTNG_UST_TRACEPOINT_EVENT_INSTANCE(hip_instrumentation, instr_activity,
                                    hip_instrumentation, queue_compute_end,
                                    LTTNG_UST_TP_ARGS(const void*, instr))

LTTNG_UST_TRACEPOINT_EVENT_INSTANCE(hip_instrumentation, instr_activity,
                                    hip_instrumentation, queue_record_begin,
                                    LTTNG_UST_TP_ARGS(const void*, instr))

LTTNG_UST_TRACEPOINT_EVENT_INSTANCE(hip_instrumentation, instr_activity,
                                    hip_instrumentation, queue_record_end,
                                    LTTNG_UST_TP_ARGS(const void*, instr))

// State recoverer

LTTNG_UST_TRACEPOINT_EVENT_INSTANCE(hip_instrumentation, instr_activity,
                                    hip_instrumentation, new_state_recoverer,
                                    LTTNG_UST_TP_ARGS(const void*, instr))

LTTNG_UST_TRACEPOINT_EVENT(
    hip_instrumentation, state_recoverer_register,
    LTTNG_UST_TP_ARGS(const void*, state_recoverer, const void*, device_ptr,
                      const void*, copy_ptr),
    LTTNG_UST_TP_FIELDS(lttng_ust_field_integer_hex(
        uintptr_t, state_recoverer,
        reinterpret_cast<uintptr_t>(state_recoverer))
                            lttng_ust_field_integer_hex(
                                uintptr_t, device_ptr,
                                reinterpret_cast<uintptr_t>(device_ptr))))

LTTNG_UST_TRACEPOINT_EVENT(hip_instrumentation, state_recoverer_cleanup,
                           LTTNG_UST_TP_ARGS(const void*, state_recoverer),
                           LTTNG_UST_TP_FIELDS(lttng_ust_field_integer_hex(
                               uintptr_t, state_recoverer,
                               reinterpret_cast<uintptr_t>(state_recoverer))))

// ----- Trace management ----- //

// Main thread

LTTNG_UST_TRACEPOINT_EVENT_CLASS(
    hip_instrumentation, trace_record,
    LTTNG_UST_TP_ARGS(const void*, instr, const void*, data, uint64_t,
                      instr_stamp),
    LTTNG_UST_TP_FIELDS(
        lttng_ust_field_integer_hex(uintptr_t, instr,
                                    reinterpret_cast<uintptr_t>(instr))
            lttng_ust_field_integer_hex(uintptr_t, data,
                                        reinterpret_cast<uintptr_t>(data))
                lttng_ust_field_integer(uint64_t, instr_stamp, instr_stamp)))

LTTNG_UST_TRACEPOINT_EVENT_INSTANCE(
    hip_instrumentation, trace_record, hip_instrumentation,
    register_thread_counters,
    LTTNG_UST_TP_ARGS(const void*, instr, const void*, data, uint64_t, stamp))

LTTNG_UST_TRACEPOINT_EVENT_INSTANCE(hip_instrumentation, trace_record,
                                    hip_instrumentation, register_wave_counters,
                                    LTTNG_UST_TP_ARGS(const void*, instr,
                                                      const void*, data,
                                                      uint64_t, stamp))

LTTNG_UST_TRACEPOINT_EVENT_INSTANCE(
    hip_instrumentation, trace_record, hip_instrumentation, register_queue,
    LTTNG_UST_TP_ARGS(const void*, instr, const void*, data, uint64_t, stamp))

LTTNG_UST_TRACEPOINT_EVENT_INSTANCE(
    hip_instrumentation, trace_record, hip_instrumentation,
    register_global_memory_queue,
    LTTNG_UST_TP_ARGS(const void*, instr, const void*, data, uint64_t, stamp))

// Collector thread

LTTNG_UST_TRACEPOINT_EVENT_CLASS(
    hip_instrumentation, collector_dump,
    LTTNG_UST_TP_ARGS(void*, ostream, const void*, data, uint64_t, instr_stamp),
    LTTNG_UST_TP_FIELDS(
        lttng_ust_field_integer_hex(uintptr_t, ostream,
                                    reinterpret_cast<uintptr_t>(ostream))
            lttng_ust_field_integer_hex(uintptr_t, data,
                                        reinterpret_cast<uintptr_t>(data))
                lttng_ust_field_integer(uint64_t, instr_stamp, instr_stamp)))

LTTNG_UST_TRACEPOINT_EVENT_INSTANCE(hip_instrumentation, collector_dump,
                                    hip_instrumentation, collector_dump_thread,
                                    LTTNG_UST_TP_ARGS(void*, ostream,
                                                      const void*, data,
                                                      uint64_t, instr_stamp))

LTTNG_UST_TRACEPOINT_EVENT_INSTANCE(hip_instrumentation, collector_dump,
                                    hip_instrumentation, collector_dump_wave,
                                    LTTNG_UST_TP_ARGS(void*, ostream,
                                                      const void*, data,
                                                      uint64_t, instr_stamp))

LTTNG_UST_TRACEPOINT_EVENT_INSTANCE(hip_instrumentation, collector_dump,
                                    hip_instrumentation, collector_dump_queue,
                                    LTTNG_UST_TP_ARGS(void*, ostream,
                                                      const void*, data,
                                                      uint64_t, instr_stamp))

LTTNG_UST_TRACEPOINT_EVENT(
    hip_instrumentation, collector_dump_end,
    LTTNG_UST_TP_ARGS(const void*, data, uint64_t, instr_stamp),
    LTTNG_UST_TP_FIELDS(
        lttng_ust_field_integer_hex(uintptr_t, data,
                                    reinterpret_cast<uintptr_t>(data))
            lttng_ust_field_integer(uint64_t, instr_stamp, instr_stamp)))

// ----- Kernel timing pass ----- //

LTTNG_UST_TRACEPOINT_EVENT(
    hip_instrumentation, kernel_timer_begin,
    LTTNG_UST_TP_ARGS(const char*, kernel, unsigned int, id),
    LTTNG_UST_TP_FIELDS(lttng_ust_field_string(kernel, kernel)
                            lttng_ust_field_integer(unsigned int, id, id)))

LTTNG_UST_TRACEPOINT_EVENT(
    hip_instrumentation, kernel_timer_end, LTTNG_UST_TP_ARGS(unsigned int, id),
    LTTNG_UST_TP_FIELDS(lttng_ust_field_integer(unsigned int, id, id)))

#endif /*_HIP_INSTRUMENTATION_TP_H */

#include <lttng/tracepoint-event.h>

#else
#define lttng_ust_tracepoint(...)
#endif
