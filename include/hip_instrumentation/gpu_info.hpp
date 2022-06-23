/** \file gpu_info.hpp
 * \brief GPU performance informations (bandwidth, FLOP/s)
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#pragma once

#include <string>
#include <vector>

/** \details A GPU information json file should look like :
 *
 * {
 *     "name":"GPU name",
 *     "memory_roofs": [
 *         {
 *             "name": "DRAM",
 *             "peak_bandwidth": 1.0
 *         },
 *         {
 *             ...
 *         }
 *     ],
 *     "compute_roofs" [
 *         {
 *             "name": "Regular FLOP",
 *             "peak_flops_s": 1.0
 *         },
 *         {
 *             ...
 *         }
 *     ]
 *
 * }
 *
 *
 */

/** \struct MemoryRoof
 * \brief Represents a peak memory bandwidth (slanted roof), in Bytes/s
 */
struct MemoryRoof {
    std::string name;
    double peak_bandwidth;
};

/** \struct ComputeRoof
 * \brief Represents a peak operational speed (straight roof), in FLOP/s
 */
struct ComputeRoof {
    /** \brief Roof name
     */
    std::string name;

    /** \brief Peak FLOP/s
     */
    double peak_flops_s;
};

/** \struct GpuInfo
 * \brief Holds information about a specific machine
 */
struct GpuInfo {
    /** \brief GPU Identifier
     */
    std::string id;

    std::vector<MemoryRoof> memory_roofs;
    std::vector<ComputeRoof> compute_roofs;

    /** \fn json
     * \brief Dump to json
     */
    std::string json() const;
};
