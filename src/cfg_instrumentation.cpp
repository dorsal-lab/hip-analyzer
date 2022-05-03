/** \file cfg_instrumentation.cpp
 * \brief
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Analysis/CFG.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Tooling/Core/Replacement.h"

#include "clang/Lex/Lexer.h"

#include <memory>
#include <sstream>
#include <string>

// Temporary
#include <iostream>

using namespace clang;
using namespace clang::ast_matchers;

namespace hip {

/** \brief Code generation
 */

std::string generateBlockCode(unsigned int id, unsigned int count) {
    std::stringstream ss;
    ss << "/* BB " << id << " (" << count << ") */" << '\n';

    ss << "_bb_counters[" << count << "][threadIdx.x] += 1;\n";

    return ss.str();
}

std::string generateInstrumentationParms() {
    std::stringstream ss;
    ss << "/* Extra params */";
    return ss.str();
}

std::string generateInstrumentationLocals(unsigned int bb_count) {
    std::stringstream ss;

    // The opening brace needs to be added to the code, in order to get "inside"
    // the kernel body. I agree that this feels like a kind of hack, but adding
    // an offset to a SourceLocation sounds tedious
    ss << "{\n/* Instrumentation locals */\n";

    ss << "__shared__ uint32_t _bb_counters[" << bb_count << "][64];\n"
       << "unsigned int _bb_count = " << bb_count << ";\n";

    // TODO (maybe) : Lexer::getIndentationForLine

    // TODO : init counters to 0

    return ss.str();
}

std::string generateInstrumentationCommit(unsigned int bb_count) {
    std::stringstream ss;

    ss << "/* Finalize instrumentation */\n";

    // Print output
    ss << "   int id = threadIdx.x;\n"
          "for (auto i = 0u; i < _bb_count; ++i) {\n"
          "printf(\" %d %d : %d\\n \", id, i, _bb_counters[i][threadIdx.x]);"
          "}\n";

    return ss.str();
}

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

        clang::tooling::Replacements reps;
        rewriter.setSourceMgr(*Result.SourceManager, lang_opt);

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
            auto error = reps.add(
                {source_manager, end_loc, 0, generateInstrumentationParms()});

            if (error) {
                throw std::runtime_error(
                    "Could not insert instrumentation extra parameters");
            }

            auto bb_count = 0u;

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
                        generateBlockCode(id, bb_count));

                    std::cout << rep.toString();
                    auto error = reps.add(rep);
                    if (error) {
                        throw std::runtime_error(
                            "Incompatible edit encountered");
                    }

                    bb_count++;
                }
            }

            /** \brief Instrumentation locals & initializations
             */

            auto body_loc = match->getBody()->getBeginLoc();
            body_loc.dump(source_manager);

            // See generateInstrumentationLocals for the explaination regarding
            // the 1 offset
            error = reps.add({source_manager, body_loc, 1,
                              generateInstrumentationLocals(bb_count)});

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
                              generateInstrumentationCommit(bb_count)});

            if (error) {
                throw std::runtime_error(
                    "Could not insert instrumentation commit block");
            }

            // Commit replacements

            applyReps(reps, rewriter);
            // rewriter.overwriteChangedFiles(); // Rewrites the input file

            rewriter.getEditBuffer(source_manager.getMainFileID())
                .write(output_file);
            output_file.close();

            // kernel = match;
        }
    }

    clang::FunctionDecl* getKernel() { return kernel; }

  private:
    std::error_code error_code;
    const std::string name;
    clang::FunctionDecl* kernel = nullptr;
    clang::Rewriter rewriter;
    llvm::raw_fd_ostream output_file;
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
