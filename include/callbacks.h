/** \file matchers.h
 * \brief AST Matcher callbacks
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#pragma once

#include "hip_instrumentation/basic_block.hpp"
#include "instr_generator.h"

#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Tooling/Core/Replacement.h"

#include <memory>

namespace hip {

// Classes

/** \class KernelCfgInstrumenter
 * \brief AST Matcher callback to instrument CFG blocks. To be run first
 */
class KernelCfgInstrumenter
    : public clang::ast_matchers::MatchFinder::MatchCallback {
  public:
    KernelCfgInstrumenter(const std::string& kernel_name,
                          std::vector<hip::BasicBlock>& b,
                          std::unique_ptr<hip::InstrGenerator> instr_gen =
                              std::make_unique<hip::InstrGenerator>());

    virtual void
    run(const clang::ast_matchers::MatchFinder::MatchResult& Result) override;

    const std::string& getOutputBuffer() const { return output_buffer; };

  protected:
    /**
     * \brief Extra parameters instrumentation
     */
    void addExtraParameters(const clang::FunctionDecl* match,
                            clang::SourceManager& source_manager,
                            clang::LangOptions& lang_opt);

    /**
     * \brief Instrumentation locals & initializations
     */
    void addLocals(const clang::FunctionDecl* match,
                   clang::SourceManager& source_manager,
                   clang::LangOptions& lang_opt);

    /**
     * \brief Instrumentation commit
     */
    void addCommit(const clang::FunctionDecl* match,
                   clang::SourceManager& source_manager,
                   clang::LangOptions& lang_opt);

  private:
    const std::string name;
    std::string output_buffer;

    clang::tooling::Replacements reps;
    clang::Rewriter rewriter;
    llvm::raw_string_ostream output_file;

    std::vector<hip::BasicBlock>& blocks;

    std::unique_ptr<hip::InstrGenerator> instr_generator;
};

/** \class KernelDuplicator
 * \brief Copies the original kernel, giving it a new name `new_kernel`
 */
class KernelDuplicator
    : public clang::ast_matchers::MatchFinder::MatchCallback {
  public:
    KernelDuplicator(const std::string& original, const std::string& new_kernel)
        : original(original), new_kernel(new_kernel), output_buffer(),
          output_file(output_buffer) {}

    virtual void
    run(const clang::ast_matchers::MatchFinder::MatchResult& Result) override;

    const std::string& getOutputBuffer() const { return output_buffer; };

  private:
    const std::string& original;
    const std::string& new_kernel;

    std::string output_buffer;
    clang::tooling::Replacements reps;
    clang::Rewriter rewriter;
    llvm::raw_string_ostream output_file;
};

/** \class KernelCallReplacer
 * \brief Replaces the call to a kernel `original` to a new one `new_kernel`
 */
class KernelCallReplacer
    : public clang::ast_matchers::MatchFinder::MatchCallback {
  public:
    KernelCallReplacer(const std::string& original,
                       const std::string& new_kernel)
        : original(original), new_kernel(new_kernel), output_buffer(),
          output_file(output_buffer) {}

    virtual void
    run(const clang::ast_matchers::MatchFinder::MatchResult& Result) override;

    const std::string& getOutputBuffer() const { return output_buffer; };

  private:
    const std::string& original;
    const std::string& new_kernel;

    std::string output_buffer;
    clang::tooling::Replacements reps;
    clang::Rewriter rewriter;
    llvm::raw_string_ostream output_file;
};

class KernelCallInstrumenter
    : public clang::ast_matchers::MatchFinder::MatchCallback {
  public:
    KernelCallInstrumenter(const std::string& kernel_name,
                           const std::vector<hip::BasicBlock>& b);

    virtual void
    run(const clang::ast_matchers::MatchFinder::MatchResult& Result) override;

    const std::string& getOutputBuffer() const { return output_buffer; };

  protected:
    /**
     * \brief Add runtime includes
     */
    void addIncludes(clang::SourceManager& source_manager,
                     clang::LangOptions& lang_opt);

    /**
     * \brief Kernel call adjustments
     */
    void addKernelCallDecoration(const clang::CUDAKernelCallExpr* match,
                                 clang::SourceManager& source_manager,
                                 clang::LangOptions& lang_opt);

  private:
    const std::string& kernel_name;
    std::unique_ptr<hip::InstrGenerator> instr_generator;

    std::string output_buffer;
    clang::tooling::Replacements reps;
    clang::Rewriter rewriter;
    llvm::raw_string_ostream output_file;
};

// Printers

std::unique_ptr<clang::ast_matchers::MatchFinder::MatchCallback>
makeFunPrinter();

std::unique_ptr<KernelCfgInstrumenter>
makeCfgInstrumenter(const std::string& name,
                    std::vector<hip::BasicBlock>& blocks);

} // namespace hip
