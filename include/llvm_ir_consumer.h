/** \file llvm_ir_consumer.hpp
 * \brief LLVM Intermediate representation handler
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#pragma once

#include "hip_instrumentation/basic_block.hpp"

#include <memory>
#include <vector>

#include "clang/Tooling/Tooling.h"

#include "clang/CodeGen/CodeGenAction.h"

std::unique_ptr<clang::tooling::ToolAction>
makeLLVMAction(const std::string& kernel_name,
               std::vector<hip::BasicBlock>& blocks);
