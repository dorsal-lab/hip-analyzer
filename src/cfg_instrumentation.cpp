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

#include <algorithm>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

using namespace clang;
using namespace clang::ast_matchers;

namespace hip {

/** \brief Utils
 */

std::pair<clang::SourceLocation, clang::SourceLocation>
findBlockLimits(clang::SourceManager& sm, const clang::CFGBlock* block) {
    BeforeThanCompare<clang::SourceLocation> comp(sm);

    auto converter = [](clang::CFGElement e) {
        return *e.getAs<clang::CFGStmt>();
    };

    std::vector<clang::SourceLocation> b_locs, e_locs;
    std::transform(
        block->begin(), block->end(), std::back_inserter(b_locs),
        [&](auto e) { return converter(e).getStmt()->getBeginLoc(); });

    std::transform(block->begin(), block->end(), std::back_inserter(e_locs),
                   [&](auto e) { return converter(e).getStmt()->getEndLoc(); });

    auto begin =
        *std::min_element(b_locs.begin(), b_locs.end(),
                          [&](auto e1, auto e2) { return comp(e1, e2); });

    auto end =
        *std::max_element(e_locs.begin(), e_locs.end(),
                          [&](auto e1, auto e2) { return comp(e1, e2); });

    return {begin, end};
}

void applyReps(clang::tooling::Replacements& reps, clang::Rewriter& rewriter) {
    if (!reps.empty()) {
        for (auto rep : reps) {
            rep.apply(rewriter);
        }
    }
}

bool isBlockInstrumentable(clang::ASTContext& context,
                           const clang::CFGBlock& block) {

    auto first = block.front();
    auto first_statement = first.getAs<clang::CFGStmt>();

    if (!first_statement.hasValue()) {
        return false;
    }

    auto terminator = block.getTerminatorStmt();

    if (block.size() > 1) {
        return true;
    }

    // If the statement is part of a for-loop condition
    if (terminator &&
        (terminator->getStmtClass() == clang::Stmt::ForStmtClass ||
         terminator->getStmtClass() == clang::Stmt::IfStmtClass)) {
        return false;
    }

    // If the successor is part of a foor loop condition (corresponds to the
    // end-loop action)
    if (block.succ_size() == 1) {
        auto succ = *block.succ_begin();
        auto succ_terminator = succ->getTerminatorStmt();
        if (succ_terminator &&
            (succ_terminator->getStmtClass() == clang::Stmt::ForStmtClass)) {
            return false;
        }
    }

    // Print if unhandled
    if (terminator) {

        auto parents = context.getParents(*terminator);
        std::cout << terminator->getStmtClassName();
    }

    return true;
}

std::string generateBlockJson(const clang::SourceManager& sm, unsigned int id,
                              const clang::SourceLocation& begin,
                              const clang::SourceLocation& end) {
    std::stringstream ss;

    ss << "{\"id\": " << id << ", \"begin\": \"" << begin.printToString(sm)
       << "\", \"end\": \"" << end.printToString(sm) << "\", \"flops\": null}";

    return ss.str();
}

std::string concatJson(const std::vector<std::string>& objects) {
    std::stringstream ss;

    ss << '[';

    for (auto& str : objects) {
        ss << str << ',';
    }

    // Remove the last comma and replace it with the closing bracket
    ss.seekp(-1, ss.cur);
    ss << ']';

    return ss.str();
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
        : name(kernel_name), output_file(output_filename, error_code) {
        instr_generator.kernel_name = kernel_name;
    }

    virtual void run(const MatchFinder::MatchResult& Result) {
        auto lang_opt = Result.Context->getLangOpts();
        auto& source_manager = *Result.SourceManager;

        rewriter.setSourceMgr(source_manager, lang_opt);

        if (const auto* match =
                Result.Nodes.getNodeAs<clang::FunctionDecl>(name)) {
            match->dump();

            std::vector<std::string> blocks_json;

            // Print First elements

            auto body = match->getBody();
            auto cfg = CFG::buildCFG(match, body, Result.Context,
                                     clang::CFG::BuildOptions());
            cfg->dump(lang_opt, true);

            for (auto block : *cfg.get()) {
                auto id = block->getBlockID();

                std::cout << "\nBlock " << id << '\n';

                // If the block terminator is a for-loop, then do not instrument
                // as this would mess with the syntax. We only need to
                // instrument the inner loop

                bool do_instrument =
                    isBlockInstrumentable(*Result.Context, *block);

                if (do_instrument) {
                    auto first = block->front();
                    auto first_statement = first.getAs<clang::CFGStmt>();

                    auto stmt = first_statement->getStmt();
                    stmt->dumpColor();

                    const auto [begin_loc, end_loc] =
                        findBlockLimits(source_manager, block);
                    begin_loc.dump(source_manager);

                    // Create replacement
                    clang::tooling::Replacement rep(
                        source_manager, begin_loc, 0,
                        instr_generator.generateBlockCode(id));

                    std::cout << rep.toString();
                    auto error = reps.add(rep);
                    if (error) {
                        throw std::runtime_error(
                            "Incompatible edit encountered : " +
                            llvm::toString(std::move(error)));
                    }

                    blocks_json.push_back(generateBlockJson(
                        source_manager, id, begin_loc, end_loc));

                    instr_generator.bb_count++;
                }
            }

            /** \brief Extra parameters instrumentation
             */

            auto last_param = match->parameters().back();

            // last_param->dump();

            // Get insertion location
            auto begin_loc = last_param->getEndLoc().getLocWithOffset(-1);
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
                    "Could not insert instrumentation extra parameters : " +
                    llvm::toString(std::move(error)));
            }

            /** \brief Instrumentation locals & initializations
             */

            auto body_loc = match->getBody()->getBeginLoc();
            // body_loc.dump(source_manager);

            error = reps.add({source_manager, body_loc.getLocWithOffset(1), 0,
                              instr_generator.generateInstrumentationLocals()});

            if (error) {
                throw std::runtime_error(
                    "Could not insert instrumentation locals : " +
                    llvm::toString(std::move(error)));
            }

            /** \brief Instrumentation commit
             */

            auto body_end_loc = match->getBody()->getEndLoc();
            // body_end_loc.dump(source_manager);

            // See generateInstrumentationLocals for the explaination regarding
            // the 1 offset
            error = reps.add({source_manager, body_end_loc, 0,
                              instr_generator.generateInstrumentationCommit()});

            if (error) {
                throw std::runtime_error(
                    "Could not insert instrumentation commit block : " +
                    llvm::toString(std::move(error)));
            }

            // Add runtime includes
            // match->getSourceRange().dump(source_manager);
            auto file_begin_loc = source_manager.getLocForStartOfFile(
                source_manager.getMainFileID());
            file_begin_loc.dump(source_manager);

            error = reps.add({source_manager, file_begin_loc, 0,
                              instr_generator.generateIncludes()});

            if (error) {
                throw std::runtime_error(
                    "Could not insert instrumentation includes : " +
                    llvm::toString(std::move(error)));
            }

            std::cout << concatJson(blocks_json) << '\n';

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
                    "Could not insert instrumentation var initializations : " +
                    llvm::toString(std::move(error)));
            }

            error = reps.add(
                {source_manager, match->getEndLoc(), 0,
                 instr_generator.generateInstrumentationLaunchParms()});
            if (error) {
                throw std::runtime_error(
                    "Could not insert instrumentation launch params : " +
                    llvm::toString(std::move(error)));
            }

            error = reps.add(
                {source_manager, match->getEndLoc().getLocWithOffset(2), 0,
                 instr_generator.generateInstrumentationFinalize()});
            if (error) {
                throw std::runtime_error(
                    "Could not insert instrumentation finalize : " +
                    llvm::toString(std::move(error)));
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
    // Only use non-instantiated template, see libASTmatchers doc
    return traverse(TK_IgnoreUnlessSpelledInSource,
                    functionDecl(hasName(kernel_name)).bind(kernel_name));
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
