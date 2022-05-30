/** \file llvm_ir_consumer.cpp
 * \brief LLVM Intermediate representation handler
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#include "llvm_ir_consumer.h"

#include "hip_instrumentation/basic_block.hpp"

#include "clang/CodeGen/CodeGenAction.h"
#include "clang/Frontend/CompilerInstance.h"

#include "llvm/Demangle/Demangle.h"
#include "llvm/IR/Module.h"

#include <optional>

// ----- Class definitions ----- //

class IRConsumer {
  public:
    /** ctor
     */
    IRConsumer(std::vector<hip::BasicBlock>& blocks);

    /** \fn run
     * \brief Run actions on the intermediate representation
     */
    void run(clang::CodeGenAction& action, const std::string& kernel_name);

    /** \fn correspondingBlock
     * \brief Returns the (front-end) basic block corresponding to the IR basic
     * block
     */
    std::optional<std::reference_wrapper<const hip::BasicBlock>>
    correspondingBlock(llvm::BasicBlock& bb);

  private:
    std::vector<hip::BasicBlock>& blocks;
};

/** \class LLVMActionWrapper
 * \brief Wrapper to run an EmitLLVMOnlyAction as a tool action
 */
class LLVMActionWrapper : public clang::tooling::ToolAction {
  public:
    LLVMActionWrapper(const std::string& k, std::vector<hip::BasicBlock>& b)
        : kernel_name(k), blocks(b) {}

    bool runInvocation(
        std::shared_ptr<clang::CompilerInvocation> Invocation,
        clang::FileManager* Files,
        std::shared_ptr<clang::PCHContainerOperations> PCHContainerOps,
        clang::DiagnosticConsumer* DiagConsumer) override {
        // Create a compiler instance to handle the actual work.
        clang::CompilerInstance Compiler(std::move(PCHContainerOps));
        Compiler.setInvocation(std::move(Invocation));
        Compiler.setFileManager(Files);

        // The FrontendAction can have lifetime requirements for Compiler or its
        // members, and we need to ensure it's deleted earlier than Compiler. So
        // we pass it to an std::unique_ptr declared after the Compiler
        // variable.
        auto llvm_action = std::make_unique<clang::EmitLLVMOnlyAction>();

        // Create the compiler's actual diagnostics engine.
        Compiler.createDiagnostics(DiagConsumer, /*ShouldOwnClient=*/false);
        if (!Compiler.hasDiagnostics())
            return false;

        Compiler.createSourceManager(*Files);

        const bool Success = Compiler.ExecuteAction(*llvm_action);

        IRConsumer consumer(blocks);
        consumer.run(*llvm_action, kernel_name);

        Files->clearStatCache();
        return Success;
    }

  private:
    std::string kernel_name;
    std::vector<hip::BasicBlock>& blocks;
};

std::unique_ptr<clang::tooling::ToolAction>
makeLLVMAction(const std::string& kernel_name,
               std::vector<hip::BasicBlock>& blocks) {
    return std::make_unique<LLVMActionWrapper>(kernel_name, blocks);
}

// ---- Utils ----- //

/** \fn contains
 * \brief Returns true if the substring substr is contained in str.
 */
bool contains(const std::string& str, const std::string& substr) {
    auto pos = str.find(substr);

    return pos != std::string::npos;
}

/** \fn findKernel
 * \brief Returns the kernel function declaration from the LLVM module
 */
llvm::Function& findKernel(llvm::Module* module,
                           const std::string& kernel_name) {
    for (auto& fun : module->functions()) {
        if (contains(fun.getName().str(), kernel_name)) {
            return fun;
        }
    }

    // Should have returned by now ..

    throw std::runtime_error("findKernel() : Kernel not found in LLVM Module");
}

bool operator<(const llvm::DebugLoc& lhs, const llvm::DebugLoc& rhs) {
    // Since we're trying to find the first each time, a non-valid DebugLoc is
    // the greatest
    if (!rhs) {
        return true;
    } else if (!lhs) {
        return false;
    } else {
        if (rhs.getLine() == lhs.getLine()) {
            return lhs.getCol() < rhs.getCol();
        } else {
            return lhs.getLine() < rhs.getLine();
        }
    }
}

/** \fn findFirstLine
 * \brief Returns the first debugLoc of a LLVM basic block
 */
llvm::DebugLoc findFirstLine(const llvm::BasicBlock& bb) {
    llvm::DebugLoc debug_loc;
    for (const auto& instr : bb) {
        if (instr.getDebugLoc() < debug_loc) {
            debug_loc = instr.getDebugLoc();
        }
    }
    return debug_loc;
}

/** \fn clipFilename
 * \brief Returns a clipped filename & position within the file without the full
 * path, such as <filename>:<line>:<col>
 */
std::string clipFilename(const std::string& filename) {
    auto last_sep = filename.rfind('/');
    if (last_sep == std::string::npos) {
        last_sep = 0;
    } else {
        // A '/' was found, we need to exclude it from the substring
        ++last_sep;
    }

    return filename.substr(last_sep);
}

/** \fn isWithinBlock
 * \brief Returns true if a llvm::DebugLoc is inside a clang-identified basic
 * block
 */
bool isWithinBlock(const llvm::DebugLoc& debug_loc, const hip::BasicBlock& bb) {
    // Text based comparison, single file. This is literally a hack.
    // TODO : find a much, much cleaner way.

    if (!debug_loc) {
        return false;
    }
    std::string loc;
    llvm::raw_string_ostream ostream(loc);
    debug_loc.print(ostream);

    auto stripped = clipFilename(loc);

    auto begin = clipFilename(*bb.begin_loc), end = clipFilename(*bb.end_loc);

    return (stripped >= begin) && (stripped <= end);
}

unsigned int countFlops(const llvm::BasicBlock& bb) { return 0u; }

// ---- Implementations ----- //

IRConsumer::IRConsumer(std::vector<hip::BasicBlock>& b) : blocks(b) {}

void IRConsumer::run(clang::CodeGenAction& action,
                     const std::string& kernel_name) {
    auto module = action.takeModule();

    /*
    for (auto& fun : module->functions()) {
        std::string name = fun.getName().str();
        llvm::errs() << name << " : " << llvm::demangle(name) << '\n';
    }
    */

    auto& kernel = findKernel(module.get(), kernel_name);

    unsigned int i = 0u;
    for (auto& bb : kernel) {
        // bb.print(llvm::errs());
        // TODO : find matching between Clang basic block and IR.

        const auto& first_instr = bb.front();
        // first_instr.print(llvm::errs());

        const auto& debug_loc = first_instr.getDebugLoc();

        llvm::errs() << '\n' << bb.getName() << " (" << i << ")\n";
        // debug_loc.print(llvm::errs());
        llvm::errs() << '\n';
        // findFirstLine(bb).print(llvm::errs());

        auto block = correspondingBlock(bb);

        if (block) {
            llvm::errs() << "Found block : " << block.value().get().id << '\n';
        }

        ++i;
    }
}

std::optional<std::reference_wrapper<const hip::BasicBlock>>
IRConsumer::correspondingBlock(llvm::BasicBlock& bb) {
    // The current method is rather limited as it returns the first match ..
    // there is no guaranteed 1:1 match between instructions and the (mostly
    // faulty) SourceLocation of the basic block.

    // TODO : fix it ?

    for (const auto& instr : bb) {
        const auto& debug_loc = instr.getDebugLoc();
        if (debug_loc) {
            for (const auto& clang_bb : blocks) {
                // Compare with frontend basic blocks. If it is within bounds,
                // we have found a match
                if (isWithinBlock(debug_loc, clang_bb)) {
                    return clang_bb;
                }
            }
        }
    }

    return std::nullopt;
}
