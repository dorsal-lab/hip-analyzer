/** \file llvm_instruction_counters.h
 * \brief LLVM Instruction counters for basic block static analysis
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#pragma once

#include "llvm/IR/Module.h"

namespace hip {

/** \class InstrCounter
 * \brief Abstract instruction counter. Provides an interface to process a basic
 * block and provide a summary
 */
class InstrCounter {
  public:
    /** \fn getCount
     * \brief Returns the counter value
     */
    virtual unsigned int getCount() const = 0;

    /** \fn operator()
     * \brief Process a basic block
     */
    virtual unsigned int operator()(llvm::BasicBlock& bb) = 0;
};

/** \class FlopCounter
 * \brief Counts the number of floating point operations in a basic block
 */
class FlopCounter : public InstrCounter {
  public:
    virtual unsigned int getCount() const override { return count; }

    virtual unsigned int operator()(llvm::BasicBlock& bb) override;

  private:
    unsigned int count = 0u;
};

/** \class LoadCounter
 * \brief Counts the number of loaded bytes in a basic block
 */
class LoadCounter : public InstrCounter {
  public:
    virtual unsigned int getCount() const override { return count; }

    virtual unsigned int operator()(llvm::BasicBlock& bb) override;

  private:
    unsigned int count = 0u;
};

/** \class StoreCounter
 * \brief Counts the number of stored bytes in a basic block
 */
class StoreCounter : public InstrCounter {
  public:
    virtual unsigned int getCount() const override { return count; }

    virtual unsigned int operator()(llvm::BasicBlock& bb) override;

  private:
    unsigned int count = 0u;
};

} // namespace hip
