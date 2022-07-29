/** \file state_recoverer.hpp
 * \brief GPU memory state recoverer to ensure deterministic kernel execution
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#pragma once

#include <map>
#include <memory>
#include <type_traits>
#include <vector>

#include "hip/hip_runtime.h"

namespace hip {

/** \struct TaggedPointer
 * \brief Type-aware tagged pointer
 */
struct TaggedPointer {
    /** ctor
     */
    template <typename T>
    TaggedPointer(const T* value_ptr, size_t array_size = 1u)
        : ptr{static_cast<const void*>(value_ptr)},
          size{array_size * sizeof(T)}, element_size{array_size} {}

    TaggedPointer(void* ptr, size_t total_size, size_t element_size = 1u)
        : ptr(ptr), size(total_size), element_size(element_size) {}

    // ----- Attributes ----- //

    const void* ptr;
    const size_t size;
    const size_t element_size;
    bool dirty = false;

    // ----- Utils ----- //

    bool isArray() const { return size > element_size; }
    auto operator<(const TaggedPointer& other) const { return ptr < other.ptr; }
};

/** \class StateRecoverer
 * \brief Records the GPU state in order to recover it for further kernel
 * executions
 */
class StateRecoverer {
  public:
    /** ctor
     */
    StateRecoverer() = default;

    // Delete copy constructor
    StateRecoverer(const StateRecoverer&) = delete;
    StateRecoverer& operator=(const StateRecoverer&) = delete;

    /** dtor
     */
    ~StateRecoverer();

    /** \fn saveState
     * \brief Records the state of the GPU memory
     */
    void saveState(const std::vector<TaggedPointer>& pointers);

    /** \fn rollback
     * \brief Recovers the state of the tainted memory zones
     */
    void rollback() const;

    /** \fn registerCallArgs
     * \brief To be called before or after the kernel call, to register that the
     * memory zones were potentially modified and thus need recovering when
     * rollback'd
     *
     * \details The function needs to be called with the exact same arguments as
     * the kernel. Some meta-programming magic will take care of determining
     * whether it is a pointer to GPU memory or not
     *
     **/
    template <typename T, typename... Args>
    void registerCallArgs(T value, Args... args) {
        registerCallArgs(value);
        registerCallArgs(args...);
    }

    template <typename T> void registerCallArgs(T* value);
    template <typename T> void registerCallArgs(T value) {}

  private:
    // ----- Utils ---- //

    /** \fn allocateElement
     * \brief Saves a single memory pointer
     */
    uint8_t* saveElement(const TaggedPointer& ptr);

    // ----- Attributes ----- //
    std::vector<std::pair<TaggedPointer, uint8_t*>> saved_values;
};

/** \class HipMemoryManager
 * \brief Memory manager (singleton) to record hip malloc / free operations
 */
class HipMemoryManager {
  public:
    HipMemoryManager(const HipMemoryManager&) = delete;
    HipMemoryManager operator=(const HipMemoryManager&) = delete;
    ~HipMemoryManager();

    /** \fn getInstance
     * \brief Returns the singleton instance
     */
    static HipMemoryManager& getInstance() {
        if (instance.get() == nullptr) {
            instance =
                std::unique_ptr<HipMemoryManager>(new HipMemoryManager());
        }

        return *instance;
    }

    /** \fn hipMalloc
     * \brief Allocates memory on the device
     */
    template <typename T> hipError_t hipMalloc(T** ptr, size_t size);

    /** \fn hipFree
     * \brief Frees allocated memory
     */
    template <typename T> hipError_t hipFree(T* ptr);

    /** \fn getTaggedPtr
     * \brief Returns the tagged pointer, if any, corresponding to an address
     */
    TaggedPointer& getTaggedPtr(void* ptr) {
        try {
            return alloc_map.at(ptr);
        } catch (std::out_of_range e) {
            throw std::runtime_error("HipMemoryManager::getTaggedPtr() : "
                                     "Unregistered pointer access");
        }
    }

  private:
    HipMemoryManager();

    hipError_t hipMallocWrapper(void** ptr, size_t size, size_t el_size);
    hipError_t hipFreeWrapper(void* ptr);

    char* so_handle;
    hipError_t (*hipMallocHandler)(void** ptr, size_t size);
    hipError_t (*hipFreeHandler)(void* ptr);

    static std::unique_ptr<HipMemoryManager> instance;
    std::map<void*, TaggedPointer> alloc_map;
};

template <typename T>
hipError_t HipMemoryManager::hipMalloc(T** ptr, size_t size) {
    return hipMallocWrapper(reinterpret_cast<void**>(ptr), size, sizeof(size));
}

template <typename T> hipError_t HipMemoryManager::hipFree(T* ptr) {
    return hipFreeWrapper(static_cast<void*>(ptr));
}

template <typename T> void StateRecoverer::registerCallArgs(T* ptr) {
    TaggedPointer& tagged_ptr =
        HipMemoryManager::getInstance().getTaggedPtr(ptr);

    uint8_t* cpu_ptr = saveElement(tagged_ptr);
    tagged_ptr.dirty = true;
    saved_values.emplace_back(tagged_ptr, cpu_ptr);
}

} // namespace hip
