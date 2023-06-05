/** \file hip_analzer_tracepoints.h
 * \brief LTTng-ust tracepoints for hip analyzer runtime
 */

#undef LTTNG_UST_TRACEPOINT_PROVIDER
#define LTTNG_UST_TRACEPOINT_PROVIDER hip_analyzer

#undef LTTNG_UST_TRACEPOINT_INCLUDE
#define LTTNG_UST_TRACEPOINT_INCLUDE "hip_analyzer_tracepoints.h"

#if !defined(_HIP_INSTRUMENTATION_TP_H) ||                                     \
    defined(LTTNG_UST_TRACEPOINT_HEADER_MULTI_READ)
#define _HIP_INSTRUMENTATION_TP_H

#include <lttng/tracepoint.h>

// ----- Memory operations ----- //

LTTNG_UST_TRACEPOINT_EVENT_CLASS(
    hip_instrumentation, memory, LTTNG_UST_TP_ARGS(void*, device_ptr),
    LTTNG_UST_TP_FIELDS(lttng_ust_field_integer_hex(size_t, device_ptr,
                                                    device_ptr)))

LTTNG_UST_TRACEPOINT_EVENT_INSTANCE(hip_instrumentation, memory,
                                    hip_instrumentation, hipMalloc,
                                    LTTNG_UST_TP_ARGS(void*, device_ptr))

LTTNG_UST_TRACEPOINT_EVENT_INSTANCE(hip_instrumentation, memory,
                                    hip_instrumentation, hipFree,
                                    LTTNG_UST_TP_ARGS(void*, device_ptr))

// ----- HIP Instrumentation handlers ----- //

LTTNG_UST_TRACEPOINT_EVENT_CLASS(
    hip_instrumentation, instr_activity, LTTNG_UST_TP_ARGS(void*, instr),
    LTTNG_UST_TP_FIELDS(lttng_ust_field_integer_hex(size_t, inst, instr)))

LTTNG_UST_TRACEPOINT_EVENT_CLASS(
    hip_instrumentation, instr_activity_ptr,
    LTTNG_UST_TP_ARGS(void*, instr, void* ptr),
    LTTNG_UST_TP_FIELDS(lttng_ust_field_integer_hex(size_t, instr, instr),
                        lttng_ust_field_integer_hex(size_t, instr, instr)))

LTTNG_UST_TRACEPOINT_EVENT_INSTANCE(hip_instrumentation, instr_activity,
                                    hip_instrumentation, new_instrumenter,
                                    LTTNG_UST_TP_ARGS(void*, instrumenter))

LTTNG_UST_TRACEPOINT_EVENT_INSTANCE(hip_instrumentation, instr_activity,
                                    hip_instrumentation, delete_instrumenter,
                                    LTTNG_UST_TP_ARGS(void*, instrumenter))

LTTNG_UST_TRACEPOINT_EVENT_INSTANCE(hip_instrumentation, instr_activity,
                                    hip_instrumentation, instr_to_device_begin,
                                    LTTNG_UST_TP_ARGS(void*, instrumenter))

LTTNG_UST_TRACEPOINT_EVENT_INSTANCE(hip_instrumentation, instr_activity_ptr,
                                    hip_instrumentation, instr_to_device_end,
                                    LTTNG_UST_TP_ARGS(void*, instrumenter,
                                                      void*, ptr))

LTTNG_UST_TRACEPOINT_EVENT_INSTANCE(hip_instrumentation, instr_activity,
                                    hip_instrumentation,
                                    instr_from_device_begin,
                                    LTTNG_UST_TP_ARGS(void*, instrumenter))

LTTNG_UST_TRACEPOINT_EVENT_INSTANCE(hip_instrumentation, instr_activity_ptr,
                                    hip_instrumentation, instr_from_device_end,
                                    LTTNG_UST_TP_ARGS(void*, instrumenter,
                                                      void*, ptr))

LTTNG_UST_TRACEPOINT_EVENT_INSTANCE(hip_instrumentation, instr_activity,
                                    hip_instrumentation, queue_compute_begin,
                                    LTTNG_UST_TP_ARGS(void*, queue_info))

LTTNG_UST_TRACEPOINT_EVENT_INSTANCE(hip_instrumentation, instr_activity,
                                    hip_instrumentation, queue_compute_end,
                                    LTTNG_UST_TP_ARGS(void*, queue_info))

LTTNG_UST_TRACEPOINT_EVENT_INSTANCE(hip_instrumentation, instr_activity,
                                    hip_instrumentation, queue_record,
                                    LTTNG_UST_TP_ARGS(void*, queue_info))

LTTNG_UST_TRACEPOINT_EVENT(
    hip_instrumentation, state_recoverer_register,
    LTTNG_UST_TP_ARGS(void*, state_recoverer, void*, device_ptr, void*,
                      copy_ptr),
    LTTNG_UST_TP_FIELDS(
        lttng_ust_field_integer_hex(size_t, state_recoverer, state_recoverer),
        lttng_ust_field_integer_hex(size_t, device_ptr, device_ptr)))

LTTNG_UST_TRACEPOINT_EVENT(hip_instrumentation, state_recoverer_cleanup,
                           LTTNG_UST_TP_ARGS(void*, state_recoverer),
                           LTTNG_UST_TP_FIELDS(lttng_ust_field_integer_hex(
                               size_t, state_recoverer, state_recoverer)))

// ----- Trace management ----- //

// Main thread

LTTNG_UST_TRACEPOINT_EVENT_CLASS(
    hip_instrumentation, trace_record,
    LTTNG_UST_TP_ARGS(void*, instr, void*, data, uint64_t, instr_stamp),
    LTTNG_UST_TP_FIELDS(lttng_ust_field_integer_hex(size_t, instr, instr),
                        lttng_ust_field_integer_hex(size_t, data, data),
                        lttng_ust_field_integer(uint64_t, instr_stamp,
                                                instr_stamp)))

LTTNG_UST_TRACEPOINT_EVENT_INSTANCE(
    hip_instrumentation, trace_record, hip_instrumentation,
    register_thread_counters, LTTNG_UST_TP_ARGS(void*, instr, void*, data))

LTTNG_UST_TRACEPOINT_EVENT_INSTANCE(hip_instrumentation, trace_record,
                                    hip_instrumentation, register_wave_counters,
                                    LTTNG_UST_TP_ARGS(void*, instr, void*, data,
                                                      uint64_t, stamp))

LTTNG_UST_TRACEPOINT_EVENT_INSTANCE(hip_instrumentation, trace_record,
                                    hip_instrumentation, register_queue,
                                    LTTNG_UST_TP_ARGS(void*, instr, void*, data,
                                                      uint64_t, stamp))

// Collector thread

LTTNG_UST_TRACEPOINT_EVENT_CLASS(
    hip_instrumentation, collector_dump,
    LTTNG_UST_TP_ARGS(void*, ostream, void*, data, uint64_t, instr_stamp),
    LTTNG_UST_TP_FIELDS(lttng_ust_field_integer_hex(size_t, ostream, ostream),
                        lttng_ust_field_integer_hex(size_t, data, data),
                        lttng_ust_field_integer(uint64_t, instr_stamp,
                                                instr_stamp)))

LTTNG_UST_TRACEPOINT_EVENT_INSTANCE(
    hip_instrumentation, collector, hip_instrumentation, collector_dump_thread,
    LTTNG_UST_TP_ARGS(void*, ostream, void*, data, uint64_t, instr_stamp))

LTTNG_UST_TRACEPOINT_EVENT_INSTANCE(
    hip_instrumentation, collector, hip_instrumentation, collector_dump_wave,
    LTTNG_UST_TP_ARGS(void*, ostream, void*, data, uint64_t, instr_stamp))

LTTNG_UST_TRACEPOINT_EVENT_INSTANCE(
    hip_instrumentation, collector, hip_instrumentation, collector_dump_queue,
    LTTNG_UST_TP_ARGS(void*, ostream, void*, data, uint64_t, instr_stamp))

#endif /*_HIP_INSTRUMENTATION_TP_H */

#include <lttng/tracepoint-event.h>
