/** \file instrumenters.h
 * \brief Preloadable instrumenters
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#include <string_view>

class GCNInstrumenter {
  public:
    virtual void addOptimizedRegAlloc() {}

    class Hooks {
        static constexpr std::string_view optimized_reg_alloc_sym =
            "amdgcn_hooks_optimized_reg_alloc";
    };
};

class WaveCountersGCNInstrumenter : public GCNInstrumenter {
  public:
    virtual void addOptimizedRegAlloc() override;
};

class WaveStateTracingGCNInstrumenter : public GCNInstrumenter {
  public:
    virtual void addOptimizedRegAlloc() override;
};
