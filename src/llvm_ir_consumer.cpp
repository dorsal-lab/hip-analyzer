/** \file llvm_ir_consumer.cpp
 * \brief LLVM Intermediate representation handler
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#include "llvm_ir_consumer.h"

#include "clang/CodeGen/CodeGenAction.h"
#include "clang/Frontend/CompilerInstance.h"

#include "llvm/Demangle/Demangle.h"
#include "llvm/IR/Module.h"

// ----- Class definitions ----- //

class IRConsumer {
  public:
    IRConsumer();

    void run(clang::CodeGenAction& action, const std::string& kernel_name);
};

/** \class LLVMActionWrapper
 * \brief Wrapper to run an EmitLLVMOnlyAction as a tool action
 */
class LLVMActionWrapper : public clang::tooling::ToolAction {
  public:
    LLVMActionWrapper(const std::string& k) : kernel_name(k) {}

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

        IRConsumer consumer;
        consumer.run(*llvm_action, kernel_name);

        Files->clearStatCache();
        return Success;
    }

  private:
    std::string kernel_name;
};

std::unique_ptr<clang::tooling::ToolAction>
makeLLVMAction(const std::string& kernel_name) {
    return std::make_unique<LLVMActionWrapper>(kernel_name);
}

// ---- Utils ----- //

bool contains(const std::string& str, const std::string& substr) {
    auto pos = str.find(substr);

    return pos != std::string::npos;
}

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

// ---- Implementations ----- //

IRConsumer::IRConsumer() {}

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

    for (auto& bb : kernel) {
        bb.print(llvm::errs());
        // TODO : find matching between Clang basic block and IR.
    }
}
