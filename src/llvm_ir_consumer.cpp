/** \file llvm_ir_consumer.cpp
 * \brief LLVM Intermediate representation handler
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#include "llvm_ir_consumer.hpp"

#include "clang/CodeGen/CodeGenAction.h"
#include "clang/Frontend/CompilerInstance.h"

#include "llvm/IR/Module.h"

// ----- Class definitions ----- //

class IRConsumer {
  public:
    IRConsumer();

    void run(clang::CodeGenAction& action);
};

/** \class LLVMActionWrapper
 * \brief Wrapper to run an EmitLLVMOnlyAction as a tool action
 */
class LLVMActionWrapper : public clang::tooling::ToolAction {
  public:
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
        consumer.run(*llvm_action);

        Files->clearStatCache();
        return Success;
    }
};

std::unique_ptr<clang::tooling::ToolAction> makeLLVMAction() {
    return std::make_unique<LLVMActionWrapper>();
}

// ---- Implementations ----- //

IRConsumer::IRConsumer() {}

void IRConsumer::run(clang::CodeGenAction& action) {
    auto module = action.takeModule();

    module->print(llvm::errs(), nullptr);
}
