/* ************************************************************************
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */
#include "stinkytofu/pipeline/Backend.hpp"

#include <vector>

#include "stinkytofu/analysis/CallGraph.hpp"
#include "stinkytofu/bindings/python/Module.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/hardware/ToolchainCaps.hpp"
#include "stinkytofu/pipeline/BackendRegistry.hpp"
#include "stinkytofu/transforms/asm/CFGBuilderPass.hpp"

namespace stinkytofu {
Backend::Backend(StinkyAsmModule& module) : module(module) {}

std::array<int, 3> Backend::getArch() const {
    return module.getArch();
}

bool Backend::runOptimization() {
    auto* pipeline = BackendRegistry::getArchPipeline(module.getArch());
    if (!pipeline || !pipeline->builder) return true;

    // Materialise caller->callee edges and side-effect Function::isCallee_
    // before any per-Function pipeline runs. Idempotent for the rocisa
    // converter flow (createFunction(name, /*isCallee=*/true) already sets
    // the bit). Load-bearing for the raw-asm recovery flow once Stage I
    // `lowering` ships CallSiteLoweringPass: the parser leaves every
    // Function with isCallee_=false, and isReturn() in CFGBuilderPass needs
    // the bit set on each callee for its terminating s_setpc_b64 to be
    // classified correctly.
    //
    // The graph itself is discarded here; future passes that need to query
    // it can re-run the analysis (it's O(instructions) and side-effect
    // idempotent).
    CallGraphAnalysis::run(module);

    // Entry Function: full per-arch kernel pipeline.
    //
    // The pipeline-builder populates `pm` with ScopeAdaptors that extract
    // named instruction-group regions (prefetch / loop body / etc.) from
    // the entry Function's single flat BB. These adaptors assert
    // `module.getFunction().size() == 1` (see ScopeAdaptor::runWholeKernel
    // / runMultiRegion / spliceBack), so this PM is entry-only by
    // construction; running it against a callee would either re-extract
    // from the entry (wrong) or trip the assertion. Callees get a separate
    // minimal pipeline below.
    {
        PassManager pm;
        if (!pipeline->builder(pm, module)) return true;
        configurePassManager(pm);
        pm.run(module.getFunction());
    }

    // Callees (functions_[1..]): minimal per-Function pipeline.
    //
    // Today this is just CFGBuilderPass so each callee body acquires a CFG
    // shape that respects the Stage G call/return semantics (relaxed
    // empty-targets assertion, no fall-through past the terminating return
    // setpc). The `passes` todo will widen this once DCE, peephole,
    // WaitCnt, MemTokenConsistency, SwPrefetch, and the DAG scheduler are
    // taught to treat isCall as a conservative barrier; at that point this
    // minimal-PM scaffolding can grow into a shared per-callee pipeline
    // (or be parameterised by the arch backend).
    std::vector<Function*> functions = module.getFunctions();
    for (size_t i = 1; i < functions.size(); ++i) {
        Function* callee = functions[i];
        if (!callee) continue;
        PassManager calleePM;
        calleePM.addPass(createCFGBuilderPass());
        configurePassManager(calleePM);
        calleePM.run(*callee);
    }

    return true;
}

void Backend::configurePassManager(PassManager& pm) {
    const auto& opts = module.getModuleOptions();

    GemmTileConfig gemmTileConfig;
    gemmTileConfig.arch = module.getArch();
    gemmTileConfig.TileA0 = opts.TileA0;
    gemmTileConfig.TileB0 = opts.TileB0;
    gemmTileConfig.TileM0 = opts.TileM0;
    gemmTileConfig.NumGRA = opts.NumGRA;
    gemmTileConfig.NumGRB = opts.NumGRB;
    gemmTileConfig.NumGRM = opts.NumGRM;
    gemmTileConfig.NumWaves = opts.WaveGroup0 * opts.WaveGroup1;
    pm.setGemmTileConfig(gemmTileConfig);

    AsmCapsConfig asmCapsConfig;
    auto msbVal = opts.VgprMsbMode;
    if (msbVal < 0 || msbVal > static_cast<int>(VgprMsbMode::Msb16)) msbVal = 0;
    asmCapsConfig.vgprMsbMode = static_cast<VgprMsbMode>(msbVal);

    // When VgprMsbMode was not set explicitly (standalone path without rocisa),
    // auto-probe using comgr if available.
    if (asmCapsConfig.vgprMsbMode == VgprMsbMode::None) {
        auto arch = module.getArch();
        GfxArchID archId = getGfxArchID(arch[0], arch[1], arch[2]);
        asmCapsConfig = ToolchainCaps::probe(archId);
    }

    pm.setAsmCapsConfig(asmCapsConfig);

    if (opts.EnableRemarks) {
        pm.getPassContext().setRemarksEnabled(true);
    }
}

}  // namespace stinkytofu
