/** \file cfg_instrumentation.cpp
 * \brief Kernel CFG Instrumentation
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#include "instr_generator.h"

#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Analysis/CFG.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Tooling/Core/Replacement.h"

#include "clang/Lex/Lexer.h"

#include <iostream>
#include <memory>
#include <sstream>
#include <string>

using namespace clang;
using namespace clang::ast_matchers;

namespace hip {

/** \brief Utils
 */

void applyReps(clang::tooling::Replacements& reps, clang::Rewriter& rewriter) {
    if (!reps.empty()) {
        for (auto rep : reps) {
            rep.apply(rewriter);
        }
    }
}

/** \brief Match callbacks
 */

/** \class KernelCfgInstrumenter
 * \brief AST Matcher callback to instrument CFG blocks. To be run first
 */
class KernelCfgInstrumenter : public MatchFinder::MatchCallback {
  public:
    KernelCfgInstrumenter(const std::string& kernel_name,
                          const std::string& output_filename)
        : name(kernel_name), output_file(output_filename, error_code) {}

    virtual void run(const MatchFinder::MatchResult& Result) {
        auto lang_opt = Result.Context->getLangOpts();
        auto& source_manager = *Result.SourceManager;

        rewriter.setSourceMgr(source_manager, lang_opt);

        if (const auto* match =
                Result.Nodes.getNodeAs<clang::FunctionDecl>(name)) {
            match->dump();
            auto body = match->getBody();
            auto cfg = CFG::buildCFG(match, body, Result.Context,
                                     clang::CFG::BuildOptions());
            cfg->dump(lang_opt, true);

            /** \brief Extra parameters instrumentation
             */

            auto last_param = match->parameters().back();

            last_param->dump();

            // Get insertion location
            auto begin_loc = last_param->getBeginLoc();
            auto end_loc =
                clang::Lexer::findNextToken(begin_loc, source_manager, lang_opt)
                    .getValue()
                    .getEndLoc();

            end_loc.dump(source_manager);

            // Generate extra code
            auto error =
                reps.add({source_manager, end_loc, 0,
                          instr_generator.generateInstrumentationParms()});

            if (error) {
                throw std::runtime_error(
                    "Could not insert instrumentation extra parameters");
            }

            // Print First elements
            for (auto block : *cfg.get()) {
                auto id = block->getBlockID();

                std::cout << "\nBlock " << id << '\n';

                auto first = block->front();
                auto first_statement = first.getAs<clang::CFGStmt>();

                if (first_statement.hasValue()) {
                    auto stmt = first_statement->getStmt();
                    auto begin_loc = stmt->getBeginLoc();
                    begin_loc.dump(source_manager);

                    // Unused, might cause issues when entering a conditional
                    // block
                    /*
                    auto rep_loc = clang::Lexer::getLocForEndOfToken(
                        begin_loc, 0, *Result.SourceManager, lang_opt);
                    rep_loc.dump(*Result.SourceManager);
                    */

                    stmt->dumpColor();

                    // Create replacement
                    clang::tooling::Replacement rep(
                        source_manager, stmt->getBeginLoc(), 0,
                        instr_generator.generateBlockCode(id));

                    std::cout << rep.toString();
                    auto error = reps.add(rep);
                    if (error) {
                        throw std::runtime_error(
                            "Incompatible edit encountered");
                    }

                    instr_generator.bb_count++;
                }
            }

            /** \brief Instrumentation locals & initializations
             */

            auto body_loc = match->getBody()->getBeginLoc();
            body_loc.dump(source_manager);

            error = reps.add({source_manager, body_loc.getLocWithOffset(1), 0,
                              instr_generator.generateInstrumentationLocals()});

            if (error) {
                throw std::runtime_error(
                    "Could not insert instrumentation locals");
            }

            /** \brief Instrumentation commit
             */

            auto body_end_loc = match->getBody()->getEndLoc();
            body_end_loc.dump(source_manager);

            // See generateInstrumentationLocals for the explaination regarding
            // the 1 offset
            error = reps.add({source_manager, body_end_loc, 0,
                              instr_generator.generateInstrumentationCommit()});

            if (error) {
                throw std::runtime_error(
                    "Could not insert instrumentation commit block");
            }

        } else if (const auto* match =
                       Result.Nodes.getNodeAs<clang::CUDAKernelCallExpr>(
                           name)) {
            match->dump();

            // For now, only the CUDA-style kernel launch is supported (like
            // kernel<<<...>>>) as parsing macros (which hipLaunchKernelGGL is)
            // with Clang is a bit of a pain. I hate C macros.

            // Set kernel geometry

            instr_generator.setGeometry(*match->getConfig(), source_manager);

            // Generate code

            auto error =
                reps.add({source_manager, match->getBeginLoc(), 0,
                          instr_generator.generateInstrumentationInit()});
            if (error) {
                throw std::runtime_error(
                    "Could not insert instrumentation var initializations");
            }

            error = reps.add(
                {source_manager, match->getEndLoc(), 0,
                 instr_generator.generateInstrumentationLaunchParms()});
            if (error) {
                throw std::runtime_error(
                    "Could not insert instrumentation launch params");
            }

            error = reps.add(
                {source_manager, match->getEndLoc().getLocWithOffset(2), 0,
                 instr_generator.generateInstrumentationFinalize()});
            if (error) {
                throw std::runtime_error(
                    "Could not insert instrumentation finalize");
            }

            // This line is (probably!) launched after the first block, so the
            // kernel instrumentation is already performed

            applyReps(reps, rewriter);
            // rewriter.overwriteChangedFiles(); // Rewrites the input file

            rewriter.getEditBuffer(source_manager.getMainFileID())
                .write(output_file);
            output_file.close();
        }
    }

  private:
    std::error_code error_code;
    const std::string name;

    clang::tooling::Replacements reps;
    clang::Rewriter rewriter;
    llvm::raw_fd_ostream output_file;

    hip::InstrGenerator instr_generator;
};

/** \class KernelCallInstrumenter
 * \brief AST Matcher for cuda kernel call
 */
class KernelCallInstrumenter : public MatchFinder::MatchCallback {
  public:
    KernelCallInstrumenter(const std::string& kernel_name,
                           const std::string& output_filename)
        : name(kernel_name), output_file(output_filename, error_code) {}

    virtual void run(const MatchFinder::MatchResult& Result) {
        auto lang_opt = Result.Context->getLangOpts();
        auto& source_manager = *Result.SourceManager;

        clang::tooling::Replacements reps;
        rewriter.setSourceMgr(source_manager, lang_opt);

        if (const auto* match =
                Result.Nodes.getNodeAs<clang::CUDAKernelCallExpr>(name)) {
            match->dump();
            /*
                        auto last_arg = match->arguments().back();
                        last_arg->dump();

                        last_arg->getLocEnd().dump(*Result.SourceManager);
            */
            match->getEndLoc().dump(source_manager);

            match->getRParenLoc().dump(source_manager);

            clang::Lexer::getLocForEndOfToken(match->getEndLoc(), 0,
                                              source_manager, lang_opt)
                .dump(source_manager);
        }
    }

  private:
    std::error_code error_code;
    const std::string name;
    clang::FunctionDecl* kernel = nullptr;
    clang::Rewriter rewriter;
    llvm::raw_fd_ostream output_file;
};

/** \brief AST matchers
 */
clang::ast_matchers::DeclarationMatcher
kernelMatcher(const std::string& kernel_name) {
    return functionDecl(hasName(kernel_name)).bind(kernel_name);
}

clang::ast_matchers::StatementMatcher
kernelCallMatcher(const std::string& kernel_name) {
    return cudaKernelCallExpr(callee(functionDecl(hasName(kernel_name))))
        .bind(kernel_name);
}

/** \brief MatchCallbacks
 */
std::unique_ptr<MatchFinder::MatchCallback>
makeCfgInstrumenter(const std::string& kernel, const std::string& output_file) {
    return std::make_unique<KernelCfgInstrumenter>(kernel, output_file);
}

std::unique_ptr<MatchFinder::MatchCallback>
makeCudaCallInstrumenter(const std::string& kernel,
                         const std::string& output_file) {
    return std::make_unique<KernelCallInstrumenter>(kernel, output_file);
}

} // namespace hip
