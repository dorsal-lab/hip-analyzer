/** \file state_recoverer.hpp
 * \brief GPU memory state recoverer to ensure deterministic kernel execution
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#pragma once

#include <map>
#include <memory>
#include <vector>

namespace hip {

/** \struct TaggedPointer
 * \brief Type-aware tagged pointer
 */
struct TaggedPointer {
    /** ctor
     */
    template <typename T>
    TaggedPointer(T* value_ptr, size_t array_size = 1u)
        : ptr{static_cast<void*>(value_ptr)}, size{array_size * sizeof(T)},
          element_size{array_size} {}

    // ----- Attributes ----- //

    const void* ptr;
    const size_t size;
    const size_t element_size;
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

    void rollback() const;

  private:
    std::map<TaggedPointer, void*> saved_values;
};

} // namespace hip
