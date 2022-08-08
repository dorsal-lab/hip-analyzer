/** \file instr_generator.cpp
 * \brief Kernel CFG Instrumentation code generation tools
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#include "instr_generator.h"

#include "clang/AST/ExprCXX.h"
#include "clang/Lex/Lexer.h"

#include <sstream>

namespace hip {

std::string getExprText(const clang::Expr* expr,
                        const clang::SourceManager& sm) {

    auto begin_loc = expr->getBeginLoc();
    auto end_loc = expr->getEndLoc().getLocWithOffset(1);

    if (begin_loc.isInvalid() || end_loc.isInvalid()) {

        throw std::runtime_error("getExprText : Could not cast expr, unhandled "
                                 "geometry declaration");
    }

    auto text = clang::Lexer::getSourceText(
                    clang::CharSourceRange::getTokenRange({begin_loc, end_loc}),
                    sm, clang::LangOptions())
                    .str();

    // Handle strange edge case where the following ',' could be included in the
    // token
    if (text.back() == ',') {
        text.pop_back();
    }
    return text;
}

void InstrGenerator::setGeometry(const clang::CallExpr& kernel_call,
                                 const clang::SourceManager& source_manager) {
    auto blocks_expr = kernel_call.getArg(0);
    blocks_expr->dump();

    blocks = getExprText(blocks_expr, source_manager);
    llvm::errs() << blocks << '\n';

    auto threads_expr = kernel_call.getArg(1);
    threads_expr->dump();

    threads = getExprText(threads_expr, source_manager);
    llvm::errs() << threads << '\n';
}

// ----- CfgCounterInstrGenerator ----- //

std::string CfgCounterInstrGenerator::generateBlockCode(unsigned int id) const {
    std::stringstream ss;
    ss << "/* BB " << id << " (" << bb_count << ") */" << '\n';

    ss << "_bb_counters[" << bb_count << "][threadIdx.x] += 1;\n";

    return ss.str();
}

std::string CfgCounterInstrGenerator::generateIncludes() const {
    return "#include \"hip_instrumentation/hip_instrumentation.hpp\"\n";
}

std::string
CfgCounterInstrGenerator::generateIncludesPost(bool rollback) const {
    if (rollback) {
        // Is there really something to add, now that we're hijacking hipMalloc?
        return "";
    } else {
        return "";
    }
}

std::string CfgCounterInstrGenerator::generateInstrumentationParms() const {
    std::stringstream ss;
    ss << ",/* Extra params */ uint8_t* _instr_ptr = nullptr";

    return ss.str();
}

std::string CfgCounterInstrGenerator::generateInstrumentationLocals() const {
    std::stringstream ss;

    ss << "\n/* Instrumentation locals */\n";

    ss << "__shared__ uint8_t _bb_counters[" << bb_count << "][64];\n"
       << "unsigned int _bb_count = " << bb_count << ";\n"
       << "#pragma unroll"
          "\nfor(auto i = 0u; i < _bb_count; ++i) { "
          "_bb_counters[i][threadIdx.x] = 0; }\n";

    return ss.str();
}

std::string CfgCounterInstrGenerator::generateInstrumentationCommit() const {
    std::stringstream ss;

    ss << "/* Finalize instrumentation */\n";

    // Print output

    ss << "    int id = threadIdx.x;\n"
          "    for (auto i = 0u; i < _bb_count; ++i) {\n"
          "        _instr_ptr[blockIdx.x * blockDim.x * _bb_count + "
          "threadIdx.x * _bb_count + i] = _bb_counters[i][threadIdx.x]\n;"
          "    }\n";

    return ss.str();
}

std::string CfgCounterInstrGenerator::generateInstrumentationInit(
    std::optional<std::string> call_args) const {
    std::stringstream ss;

    ss << "/* Instrumentation variables, hipMalloc, etc. */\n\n";

    if (call_args.has_value()) {
        ss << "hip::StateRecoverer _" << kernel_name << "_recoverer;\n"
           << "_" << kernel_name << "_recoverer.registerCallArgs("
           << call_args.value() << ");\n";
    }

    ss << "hip::KernelInfo _" << kernel_name << "_info(\"" << kernel_name
       << "\", " << bb_count << ", " << blocks << ", " << threads << ");\n";

    ss << "hip::Instrumenter _" << kernel_name << "_instr(_" << kernel_name
       << "_info);\n";

    ss << "auto _" << kernel_name << "_ptr = _" << kernel_name
       << "_instr.toDevice();\n\n";

    return ss.str();
}

std::string
CfgCounterInstrGenerator::generateInstrumentationLaunchParms() const {
    std::stringstream ss;

    ss << ",/* Extra parameters for kernel launch ( " << bb_count
       << " )*/ (uint8_t*) _" << kernel_name << "_ptr";

    return ss.str();
}

std::string
CfgCounterInstrGenerator::generateInstrumentationFinalize(bool rollback) const {
    std::stringstream ss;

    ss << "\n\n/* Finalize instrumentation : copy back data */\n";

    // Fetch data from device

    ss << "hip::check(hipDeviceSynchronize());\n"
       << "_" << kernel_name << "_instr.fromDevice(_" << kernel_name
       << "_ptr);\n";

    ss << "_" << kernel_name << "_instr.record();\n";

    // Free memory

    ss << "hip::check(hipFree(_" << kernel_name << "_ptr));\n\n";

    if (rollback) {
        ss << "_" << kernel_name << "_recoverer.rollback();\n";
    }

    return ss.str();
}

// ----- EventRecordInstrGenerator ----- //

std::string
EventRecordInstrGenerator::generateBlockCode(unsigned int id) const {
    std::stringstream ss;
    ss << "/* BB " << id << " (" << bb_count << ") */" << '\n';

    ss << "_queue.push_back({" << bb_count << "});\n";

    return ss.str();
}

std::string EventRecordInstrGenerator::generateIncludes() const {
    return "#include \"hip_instrumentation/hip_instrumentation.hpp\"\n"
           "#include \"hip_instrumentation/gpu_queue.hpp\"\n"
           "#include \"hip_instrumentation/state_recoverer.hpp\"\n";
}

std::string
EventRecordInstrGenerator::generateIncludesPost(bool rollback) const {
    return "";
}

std::string EventRecordInstrGenerator::generateInstrumentationParms() const {
    std::stringstream ss;

    // Must have default parameters overload, otherwise the kernel call won't
    // compile
    ss << ",/* Extra params */ hip::Event* _event_storage = nullptr, size_t* "
          "_event_offsets = nullptr";

    return ss.str();
}

std::string EventRecordInstrGenerator::generateInstrumentationLocals() const {
    std::stringstream ss;

    ss << "\n/* Instrumentation locals */\n";

    if (is_thread) {
        ss << "hip::ThreadQueue<hip::Event> _queue{_event_storage, "
              "_event_offsets};\n";
    } else {
        ss << "hip::WaveQueue<hip::Event> _queue{_event_storage, "
              "_event_offsets};\n";
    }

    return ss.str();
}

std::string EventRecordInstrGenerator::generateInstrumentationCommit() const {
    std::stringstream ss;

    ss << "/* Finalize instrumentation */\n";

    // Print output

    // Nothing to do : the queue writes to global memory

    return ss.str();
}

std::string EventRecordInstrGenerator::generateInstrumentationInit(
    std::optional<std::string> call_args) const {
    std::stringstream ss;

    ss << "/* Instrumentation variables, hipMalloc, etc. */\n\n";

    if (call_args.has_value()) {
        // The call args MUST have already been recorded in order to launch an
        // event record instrumentation.
    }

    // Just as kernel info and instrumenter must already exist

    // TODO : pass parameters for wave queue

    ss << "auto _queue_info = hip::QueueInfo::thread<hip::Event>(_"
       << kernel_name << "_instr);\n";

    ss << "auto _event_storage = _queue_info.allocBuffer<hip::Event>();\n"
       << "auto _event_offsets = _queue_info.allocOffsets();\n";

    return ss.str();
}

std::string
EventRecordInstrGenerator::generateInstrumentationLaunchParms() const {
    std::stringstream ss;

    ss << ",/* Extra parameters for kernel launch ( " << bb_count
       << " )*/ _event_storage, _event_offsets";

    return ss.str();
}

std::string EventRecordInstrGenerator::generateInstrumentationFinalize(
    bool rollback) const {
    std::stringstream ss;

    ss << "\n\n/* Finalize instrumentation : copy back data */\n";

    // TODO

    // Free device memory

    ss << "hip::check(hipFree(_event_storage));\n"
          "hip::check(hipFree(_event_offsets));\n";

    if (rollback) {
        ss << "_" << kernel_name << "_recoverer.rollback();\n";
    }

    return ss.str();
}

}; // namespace hip
