/** \file actions_processor.h
 * \brief Processor to chain different compiler actions and produce a single
 * output
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#include "actions_processor.h"

#include <fstream>
#include <iostream>
#include <sstream>

#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Rewrite/Core/Rewriter.h"

#include "callbacks.h"
#include "llvm_ir_consumer.h"
#include "matchers.h"

namespace hip {

ActionsProcessor::ActionsProcessor(const std::string& input_file,
                                   const clang::tooling::CompilationDatabase& d,
                                   const std::string& output_file_path)
    : input_file_path(clang::tooling::getAbsolutePath(input_file)), db(d),
      output_file(output_file_path) {

    std::stringstream ss;
    ss << std::ifstream(input_file).rdbuf();
    buffer = ss.str();
}

ActionsProcessor::~ActionsProcessor() {
    std::ofstream out(output_file);
    out << buffer;
    out.close();
}

ActionsProcessor& ActionsProcessor::process(
    std::function<std::string(clang::tooling::ClangTool&)> action) {
    clang::tooling::ClangTool tool(db, {input_file_path});

    tool.mapVirtualFile(input_file_path, buffer);

    auto ret = action(tool);

    if (!ret.empty()) {
        buffer = std::move(ret);
    }

    return *this;
}

ActionsProcessor& ActionsProcessor::observe(
    std::function<void(clang::tooling::ClangTool&)> action) {
    clang::tooling::ClangTool tool(db, {input_file_path});

    tool.mapVirtualFile(input_file_path, buffer);

    action(tool);

    return *this;
}

ActionsProcessor& ActionsProcessor::observeOriginal(
    std::function<void(clang::tooling::ClangTool&)> action) {
    clang::tooling::ClangTool tool(db, {input_file_path});

    action(tool);

    return *this;
}

namespace actions {

std::unique_ptr<InstrGenerator> make_tracer(TraceType trace_type) {
    switch (trace_type) {
    case TraceType::Event:
        return std::make_unique<hip::EventRecordInstrGenerator>();

    case TraceType::TaggedEvent:
        return std::make_unique<hip::EventRecordInstrGenerator>(
            true, "hip::TaggedEvent");

    case TraceType::WaveState:
        return std::make_unique<hip::EventRecordInstrGenerator>(
            false, "hip::WaveState");

    case TraceType::None:
        throw std::runtime_error(
            "hip::actions::TraceBasicBlocks::() : Logic error : tracing, but "
            "TraceType::None received");
    }
}

std::string DuplicateKernel::operator()(clang::tooling::ClangTool& tool) {
    // Kernel matcher
    clang::ast_matchers::MatchFinder finder;
    auto kernel_matcher = hip::kernelMatcher(original);

    hip::KernelDuplicator duplicator(original, new_kernel, rollback);

    finder.addMatcher(kernel_matcher, &duplicator);

    err |= tool.run(clang::tooling::newFrontendActionFactory(&finder).get());
    return duplicator.getOutputBuffer();
}

std::string InstrumentBasicBlocks::operator()(clang::tooling::ClangTool& tool) {
    // Kernel matcher
    clang::ast_matchers::MatchFinder finder;
    auto kernel_matcher = hip::kernelMatcher(kernel);

    // Instrument basic blocks
    auto kernel_instrumenter = hip::makeCfgInstrumenter(kernel, blocks);

    /* auto kernel_call_instrumenter = hip::makeCudaCallInstrumenter(
    kernel_name.getValue(), output_file.getValue()); */

    finder.addMatcher(kernel_matcher, kernel_instrumenter.get());
    err |= tool.run(clang::tooling::newFrontendActionFactory(&finder).get());
    return kernel_instrumenter->getOutputBuffer();
}

std::string InstrumentBasicBlocks::getInstrumentedKernelName(
    const std::string& kernel_name) {
    return kernel_name + "_bb";
}

std::string TraceBasicBlocks::operator()(clang::tooling::ClangTool& tool) {
    // Kernel matcher
    clang::ast_matchers::MatchFinder finder;
    auto kernel_matcher = hip::kernelMatcher(kernel);

    // Instrument basic blocks
    auto tracing_instr_generator = make_tracer(trace_type);

    auto kernel_instrumenter = std::make_unique<hip::KernelCfgInstrumenter>(
        kernel, blocks, std::move(tracing_instr_generator));

    /* auto kernel_call_instrumenter = hip::makeCudaCallInstrumenter(
    kernel_name.getValue(), output_file.getValue()); */

    finder.addMatcher(kernel_matcher, kernel_instrumenter.get());
    err |= tool.run(clang::tooling::newFrontendActionFactory(&finder).get());
    return kernel_instrumenter->getOutputBuffer();
}

std::string
TraceBasicBlocks::getInstrumentedKernelName(const std::string& kernel_name) {
    return kernel_name + "_traced";
}

std::string AnalyzeIR::operator()(clang::tooling::ClangTool& tool) {
    auto codegen = makeLLVMAction(kernel, blocks);
    err |= tool.run(codegen.get());
    return "";
}

std::string ReplaceKernelCall::operator()(clang::tooling::ClangTool& tool) {
    clang::ast_matchers::MatchFinder finder;
    auto kernel_call_matcher = hip::kernelCallMatcher(original);

    KernelCallReplacer replacer(original, new_kernel);

    finder.addMatcher(kernel_call_matcher, &replacer);

    err |= tool.run(clang::tooling::newFrontendActionFactory(&finder).get());
    return replacer.getOutputBuffer();
}

std::string InstrumentKernelCall::operator()(clang::tooling::ClangTool& tool) {
    clang::ast_matchers::MatchFinder finder;
    auto kernel_call_matcher = hip::kernelCallMatcher(instrumented_kernel);

    KernelCallInstrumenter instrumenter(kernel, instrumented_kernel, blocks,
                                        do_rollback);

    finder.addMatcher(kernel_call_matcher, &instrumenter);

    err |= tool.run(clang::tooling::newFrontendActionFactory(&finder).get());
    return instrumenter.getOutputBuffer();
}

std::string TraceKernelCall::operator()(clang::tooling::ClangTool& tool) {
    clang::ast_matchers::MatchFinder finder;
    auto kernel_call_matcher = hip::kernelCallMatcher(instrumented_kernel);

    auto instr_generator = make_tracer(trace_type);

    KernelCallInstrumenter instrumenter(kernel, instrumented_kernel, blocks,
                                        true, std::move(instr_generator));

    finder.addMatcher(kernel_call_matcher, &instrumenter);

    err |= tool.run(clang::tooling::newFrontendActionFactory(&finder).get());
    return instrumenter.getOutputBuffer();
}

std::string DuplicateKernelCall::operator()(clang::tooling::ClangTool& tool) {
    clang::ast_matchers::MatchFinder finder;
    auto kernel_call_matcher = hip::kernelCallMatcher(original);

    KernelCallDuplicator replacer(original, new_kernel);

    finder.addMatcher(kernel_call_matcher, &replacer);

    err |= tool.run(clang::tooling::newFrontendActionFactory(&finder).get());
    return replacer.getOutputBuffer();
}

} // namespace actions

} // namespace hip
