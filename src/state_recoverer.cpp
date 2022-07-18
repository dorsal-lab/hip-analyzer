/** \file state_recoverer.cpp
 * \brief GPU memory state recoverer to ensure deterministic kernel execution
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#include "hip_instrumentation/state_recoverer.hpp"

namespace hip {

StateRecoverer::~StateRecoverer() {}

void StateRecoverer::saveState(const std::vector<TaggedPointer>& pointers) {}

void StateRecoverer::rollback() const {}

} // namespace hip
