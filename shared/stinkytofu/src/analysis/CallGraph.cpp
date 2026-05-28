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
#include "stinkytofu/analysis/CallGraph.hpp"

#include "stinkytofu/bindings/python/Module.hpp"
#include "stinkytofu/core/BasicBlock.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/core/IRBase.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/support/Casting.hpp"

namespace stinkytofu {

CallGraph CallGraphAnalysis::run(StinkyAsmModule& module) {
    CallGraph result;

    // functions_[0] is the entry by the StinkyAsmModule contract. We never
    // flip isCallee_ on the entry Function even if it shows up as a callee
    // candidate -- see the class docstring for the recursion rationale.
    std::vector<Function*> functions = module.getFunctions();
    Function* entry = functions.empty() ? nullptr : functions.front();

    for (Function* caller : functions) {
        if (!caller) continue;
        for (BasicBlock& bb : *caller) {
            for (IRBase& irNode : bb) {
                auto* inst = dyn_cast<StinkyInstruction>(&irNode);
                if (!inst || !isCall(*inst)) continue;

                // Opaque indirect call: no CallSiteData attached. The site
                // is real but contributes zero edges to the graph. Pass-side
                // consumers still see it via isCall(*inst).
                const std::vector<std::string>& names = getCalleeFunctions(*inst);
                if (names.empty()) continue;

                std::vector<CallGraphEdge>& outs = result.outEdges[caller];
                outs.reserve(outs.size() + names.size());
                for (const std::string& calleeName : names) {
                    Function* callee = module.getFunction(calleeName);
                    if (!callee) result.hasUnresolvedCallees = true;
                    outs.push_back(CallGraphEdge{inst, callee, calleeName});
                    if (!callee) continue;

                    result.inEdges[callee].emplace_back(caller, inst);

                    // Side-effect: flip isCallee_ so a terminating
                    // s_setpc_b64 in this Function is recognised by
                    // isReturn(). Suppressed for the entry Function (see
                    // class docstring).
                    if (callee != entry) callee->setIsCallee(true);
                }
            }
        }
    }

    return result;
}

}  // namespace stinkytofu
