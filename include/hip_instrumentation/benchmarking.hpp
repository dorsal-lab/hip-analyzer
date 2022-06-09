/** \file benchmarking.hpp
 * \brief Roofline benchmarking utilities
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#pragma once

namespace hip {

namespace benchmarking {

/** \fn benchmarkFlops
 * \brief Return the maximum floating point operations per second on the device,
 * using only "traditionnal" operations (no FMA / tensor cores)
 */
float benchmarkFlops();

/** \fn benchmarkGlobalMemory
 * \brief Returns the maximum bandwidth (Bytes/s) for global memory access, with
 * stride-1 access
 */
float benchmarkGlobalMemory();

} // namespace benchmarking

} // namespace hip
