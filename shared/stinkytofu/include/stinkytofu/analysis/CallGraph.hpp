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
#pragma once

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "stinkytofu/Export.hpp"

namespace stinkytofu {
class Function;
class StinkyAsmModule;
struct StinkyInstruction;

/// One directed edge in the call graph (caller -> callee).
///
/// A static call site (CallSiteData::calleeFuncs.size() == 1) contributes one
/// edge. A runtime-dispatched site (size N > 1) contributes N edges that all
/// share the same `callSite` pointer; exactly one is taken at runtime based
/// on the address the caller materialised into the swappc source SGPR pair.
///
/// `callee` is nullptr when the name in CallSiteData::calleeFuncs does not
/// resolve to any Function in the module. The edge is still recorded with
/// its original `calleeName` so a verifier / debug dump can surface the
/// dangling reference; consumers that walk the graph must null-check
/// `callee` before dereferencing.
struct CallGraphEdge {
    StinkyInstruction* callSite;  ///< The s_swappc_b64 in the caller Function.
    Function* callee;             ///< Resolved callee Function, or nullptr.
    std::string calleeName;       ///< Original name from CallSiteData::calleeFuncs.
};

/// Result of CallGraphAnalysis for a single StinkyAsmModule. Both views
/// (caller-keyed `outEdges` and callee-keyed `inEdges`) are populated in
/// one pass so consumers can pick whichever direction they need without
/// re-walking the IR.
///
/// Edge addresses are NOT stable across runs: rebuilding the analysis
/// invalidates any pointers held into the previous CallGraph.
struct CallGraph {
    /// For each Function that contains at least one call site, the list of
    /// out-edges (caller -> {callSite, callee, name}). Functions with no
    /// call sites are absent from the map (lookup returns end()).
    std::unordered_map<Function*, std::vector<CallGraphEdge>> outEdges;

    /// For each callee Function reached by at least one edge, the list of
    /// (caller, callSite) pairs that target it. Useful for "who calls this?"
    /// queries (e.g. dead-callee detection, inlining heuristics).
    std::unordered_map<Function*, std::vector<std::pair<Function*, StinkyInstruction*>>> inEdges;

    /// True if any call-site name in the module failed to resolve to a
    /// Function. The Stage K test todo will assert this is false for the
    /// rocisa-fed path; the raw-asm recovery path (Stage I `lowering`)
    /// must run CallSiteLoweringPass before relying on this.
    bool hasUnresolvedCallees = false;
};

/// Module-scope analysis that builds the call graph for a StinkyAsmModule
/// and, as a side effect, marks every Function reached by at least one
/// edge as a callee via Function::setIsCallee(true).
///
/// Scope rationale
/// ---------------
/// Unlike per-Function analyses (BBIndexAnalysis, DominanceAnalysis,
/// LoopAnalysis) which register with the per-Function AnalysisManager, the
/// call graph requires visibility across every Function in the module to
/// resolve callee names. So it lives outside that machinery and is NOT
/// added to registerAllAnalyses(). The Backend (Stage I, `pipeline`) calls
/// run(...) once at the start of optimisation, before iterating any
/// per-Function PassManagers.
///
/// Side effect on Function::isCallee_
/// ----------------------------------
/// For the rocisa-fed flow the converter already sets isCallee_=true on
/// every Function it creates via createFunction(name, /*isCallee=*/true),
/// so this side effect is idempotent. The case it matters is the raw-asm
/// recovery flow: the parser creates every Function with the default
/// isCallee_=false; running CallGraphAnalysis after CallSiteLoweringPass
/// (Stage I, `lowering`) flips the bit on each Function reached by an
/// edge, so isReturn(setpc) in those Functions correctly classifies their
/// terminating s_setpc_b64 as a return.
///
/// The entry Function (functions_[0]) is never marked callee even if it
/// appears as a candidate in some CallSiteData::calleeFuncs list. That
/// would represent recursion into the kernel entry, which is not a
/// supported shape today; the edge is still recorded (for visibility) but
/// the isCallee_ side effect is suppressed.
///
/// Idempotency
/// -----------
/// Calling run(...) twice on the same module returns equivalent graphs and
/// leaves Function::isCallee_ in the same state (setIsCallee(true) is
/// monotonic; no Function ever has the bit cleared by this analysis).
class STINKYTOFU_EXPORT CallGraphAnalysis {
   public:
    using Result = CallGraph;

    /// Walk every Function in \p module, collecting one CallGraphEdge per
    /// candidate callee at every StinkyInstruction satisfying isCall(*inst).
    static Result run(StinkyAsmModule& module);
};

}  // namespace stinkytofu
