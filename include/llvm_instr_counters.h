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

namespace MemType {

enum class MemType {
    Floating = 0b001, // All floating point types (float, double, ...)
    Integer = 0b010,  // All integer types
    Other = 0b100     // All others : pointers, .. (?)
};

inline MemType operator|(MemType lhs, MemType rhs) {
    return static_cast<MemType>(
        static_cast<std::underlying_type_t<MemType>>(lhs) |
        static_cast<std::underlying_type_t<MemType>>(rhs));
}
inline bool operator&(MemType lhs, MemType rhs) {
    return static_cast<std::underlying_type_t<MemType>>(lhs) &
           static_cast<std::underlying_type_t<MemType>>(rhs);
}

inline bool operator==(MemType lhs, MemType rhs) {
    return static_cast<std::underlying_type_t<MemType>>(lhs) ==
           static_cast<std::underlying_type_t<MemType>>(rhs);
}

static MemType All = MemType::Floating | MemType::Integer |
                     MemType::Other; // All type of memory access

} // namespace MemType

/** \class LoadCounter
 * \brief Counts the number of loaded bytes in a basic block
 */
class LoadCounter : public InstrCounter {
  public:
    virtual unsigned int getCount() const override { return counted; }

    virtual unsigned int operator()(llvm::BasicBlock& bb) override {
        return count(bb);
    };

    unsigned int count(llvm::BasicBlock& bb,
                       MemType::MemType type_filter = MemType::All);

  private:
    unsigned int counted = 0u;
};

/** \class StoreCounter
 * \brief Counts the number of stored bytes in a basic block
 */
class StoreCounter : public InstrCounter {
  public:
    virtual unsigned int getCount() const override { return counted; }

    virtual unsigned int operator()(llvm::BasicBlock& bb) override {
        return count(bb);
    };

    unsigned int count(llvm::BasicBlock& bb,
                       MemType::MemType type_filter = MemType::All);

  private:
    unsigned int counted = 0u;
};

} // namespace hip
