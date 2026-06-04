# RFC: Autotuning for hipDNN

**Status**: Draft
**Authors**: Chris Miserva

## Table of Contents

1. [Summary](#1-summary)
2. [Motivation](#2-motivation)
3. [Requirements](#3-requirements)
4. [Terminology](#4-terminology)
5. [Architecture](#5-architecture)
6. [API Design](#6-api-design)
   - [6.1 Core Types](#61-core-types)
   - [6.2 Engine Discovery and Plan Spec Collection](#62-engine-discovery-and-plan-spec-collection)
     - [6.2.1 Concepts: Graph, Engine Configs, and Execution Plans](#621-concepts-graph-engine-configs-and-execution-plans)
     - [6.2.2 Types](#622-types)
     - [6.2.3 API on Graph](#623-api-on-graph)
     - [6.2.4 Plan Spec Collection Semantics](#624-plan-spec-collection-semantics)
     - [6.2.5 What Happens Under the Hood](#625-what-happens-under-the-hood)
     - [6.2.6 Typical Flow](#626-typical-flow)
   - [6.3 Autotuning API on Graph](#63-autotuning-api-on-graph)
   - [6.4 Benchmarking Flows](#64-benchmarking-flows)
   - [6.5 Config File Output](#65-config-file-output)
   - [6.6 Checkpoint/Resume](#66-checkpointresume)
7. [Porting Guide: cuDNN to hipDNN](#7-porting-guide-cudnn--hipdnn)
8. [Complete Example](#8-complete-example)
9. [Risks and Mitigations](#9-risks-and-mitigations)
- [Appendix A: Porting Guide from cuDNN Autotuning to hipDNN](#appendix-a-porting-guide-from-cudnn-autotuning-to-hipdnn)
  - [A.1 Auto Workflow Comparison Examples](#a1-auto-workflow-comparison-examples)
  - [A.2 Step-by-Step API Mapping](#a2-step-by-step-api-mapping)
  - [A.3 Key Differences and Similarities](#a3-key-differences-and-similarities)
- [Appendix B: Use Cases](#appendix-b-use-cases)

---

## 1. Summary

This RFC defines the autotuning system for hipDNN. Autotuning benchmarks execution plans (engine configurations) for a given operation graph on the current hardware and selects the fastest one, replacing heuristic-only engine selection with empirically measured performance data.

The autotuning runtime lives in `hipdnn_frontend` and persists results to **engine override configuration files** in the same JSON format used by `HIPDNN_ENGINE_OVERRIDE_FILE`. On subsequent runs, `EngineOverrideConfig` loads these files and selects the autotuned engine without re-benchmarking.

Runtime caching is out of scope; a general caching facility will be addressed in a future RFC.

---

## 2. Motivation

Deep learning operations can be computed by many algorithms, each with different performance characteristics depending on input dimensions, data types, workspace memory, and GPU architecture. No single algorithm is universally fastest. Heuristic engine ordering may be inaccurate for unusual configurations, new hardware, or corner cases. Autotuning addresses this by empirically measuring each candidate.

---

## 3. Requirements

1. **Two tuning modes**: Auto-tune (simple wall-time) and Exhaustive-tune (internal priming via temporary plans + wall-time)
2. **Config file output**: Write ranked winners to engine override JSON files, enabling reuse via `HIPDNN_ENGINE_OVERRIDE_FILE`
3. **Benchmarking strategies**: Single-shot, fixed-average, run-until-stable
4. **Separated phases**: Inspect/filter engines between discovery and plan spec collection
5. **Knob variant autotuning**: Benchmark the same engine with different knob configurations
6. **cuDNN API parity**: Match cuDNN's autotuning support, then extend
7. **Extensible ranking**: User-provided ranking function for custom criteria
8. **Checkpoint/resume**: Resume interrupted autotuning sessions

---

## 4. Terminology

| Term | Meaning |
|------|---------|
| **Auto-tune mode** | Benchmark engines using standard execution (no special flags). |
| **Exhaustive-tune mode** | Benchmark with engine-internal cache priming. `autotune()` builds temporary priming plans using each plan spec's knob settings plus `global.benchmarking=1`, executes them once to prime engine caches, discards them, then compiles the real plans against the now-primed engine state before timed runs. |
| **Benchmarking knob** | The `global.benchmarking` knob: a per-engine flag that tells the engine to initialize its internal cache state (e.g., MIOpen's `find` phase). Not all engines support this. This knob is NOT user-settable; it is managed exclusively by `autotune()` in EXHAUSTIVE mode. `add_engine_*()` functions reject/strip it. |
| **Graph signature** | Deterministic hash of operation graph structure, tensor shapes, data types, and compute configuration. |
| **Device signature** | Deterministic key derived from GPU properties (architecture, CU count, memory). |
| **Engine config** | A lightweight descriptor of an available engine implementation for a graph. Represented by `EngineConfigInfo`, which contains engine ID, name, available knobs, workspace size, and whether the engine supports exhaustive benchmarking. Returned by `get_engine_configs()`. |
| **Plan spec** | An engine ID paired with specific knob settings. The composite key `(engineId, knobSettings)` uniquely identifies an autotuning candidate. Stored internally by `add_engine_*()` calls. |
| **Engine variant** | An `EngineVariant` struct pairing an engine ID with specific knob settings. The user-facing type for explicit plan spec construction. |

---

## 5. Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                    hipdnn_frontend (header-only)                     │
│                                                                      │
│  ┌─────────────────┐    ┌──────────────────────────────────────┐     │
│  │ Autotuning      │    │ Config File Writer                   │     │
│  │ Runtime         │───▶│ (EngineOverrideConfig JSON format)   │     │
│  │                 │    │                                      │     │
│  │ • AUTO mode     │    │ • Write ranked winners               │     │
│  │ • EXHAUSTIVE    │    │ • Append to existing file            │     │
│  │   mode          │    │ • Replace matching (op, tensors)     │     │
│  │ • Strategies    │    │ • Autotune metadata                  │     │
│  │ • Checkpoint    │    └──────────────────────────────────────┘     │
│  └─────────────────┘                                                 │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐    │
│  │ EngineOverrideConfig (existing)                              │    │
│  │                                                              │    │
│  │ • loadFromEnv() - reads HIPDNN_ENGINE_OVERRIDE_FILE          │    │
│  │ • matchOperation(op, tensors) - O(1) exact + wildcard scan   │    │
│  │ • Returns MatchResult {engineId, knobs} for matching ops     │    │
│  └──────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────┘
```

> **Future**: RFC-0007 defines a `SelectionHeuristic::Config` plugin that would subsume `EngineOverrideConfig` lookup. Autotuning output will continue using the same JSON format or a centralized cache mechanism from a future RFC.

---

## 6. API Design

### 6.1 Core Types

```cpp
enum class TuneMode {
    AUTO,        // Simple wall-time comparison (no engine-internal cache priming)
    EXHAUSTIVE   // Build temporary priming plans internally, prime engine caches, then wall-time comparison
};

enum class AutotuneStrategy {
    SINGLE_SHOT,       // 1 timed run, take the result
    FIXED_AVERAGE,     // Average of N runs (default)
    RUN_UNTIL_STABLE   // Run until timing variance stabilizes, up to a cap
};

// User-provided ranking comparator (optional)
using AutotuneRankingFn = std::function<void(std::vector<AutotuneResult>&)>;

struct AutotuneConfig {
    TuneMode mode = TuneMode::AUTO;
    AutotuneStrategy strategy = AutotuneStrategy::FIXED_AVERAGE;

    // Warmup
    int warmupIterations = 3;

    // FIXED_AVERAGE parameter
    int timedIterations = 10;

    // RUN_UNTIL_STABLE parameters
    // Validation: windowSize >= 2, maxIterations >= windowSize,
    // stabilityThreshold must be in range (0.0, 1.0) exclusive.
    // autotune() returns an error if any of these constraints are violated.
    int maxIterations = 100;
    int windowSize = 5;
    float stabilityThreshold = 0.05f;      // Coefficient of variation threshold (5%)

    // Workspace limit (0 = no limit; skip plans exceeding this)
    size_t maxWorkspaceBytes = 0;

    // Engine filter: filters the added plan specs (empty = all engines).
    // Does not discover or add new specs. A warning is logged for engine IDs
    // in this list that have no plan specs.
    std::vector<int64_t> engineIdFilter = {};

    // Custom ranking (nullptr = rank by minTimeMs)
    AutotuneRankingFn rankingFn = nullptr;

    // EXHAUSTIVE mode: abort on priming failure (default) or continue without priming.
    // When false: autotune() returns an error if any engine's priming plan fails to build or execute.
    // When true: the engine is benchmarked without priming; AutotuneResult::ranExhaustive is set
    // to false for that engine, and errorMessage notes the priming failure reason.
    // Has no effect in AUTO mode.
    bool continueOnPrimingFailure = false;

    // Checkpoint/resume (empty = no checkpointing)
    std::filesystem::path checkpointFile = {};
};

struct AutotuneStorageConfig {
    // Config file output:
    std::filesystem::path filePath;
    bool deleteAllExistingFileContent = false;  // When true, delete all existing file content before writing
};

struct AutotuneResult {
    int rank;                               // 0-based (0 = fastest); -1 for failed engines
    int64_t engineId;
    std::vector<KnobSetting> knobSettings;  // Composite key with engineId
    std::string engineName;
    float minTimeMs;                        // Used for ranking
    float avgTimeMs;
    float stddevMs;                         // 0.0 for SINGLE_SHOT
    int iterationsRun;                      // Actual iterations executed
    bool converged;                         // true for SINGLE_SHOT and FIXED_AVERAGE (measurement
                                            // completed as requested). false only for RUN_UNTIL_STABLE
                                            // when maxIterations was reached without convergence.
    int64_t workspaceSize;
    bool succeeded;
    std::string errorMessage;
    TuneMode modeUsed;
    bool ranExhaustive;                     // true if this engine was primed via a temporary
                                            // benchmarking plan before timing. false if the engine
                                            // does not support exhaustive priming or AUTO mode was used.
};
```

### 6.2 Engine Discovery and Plan Spec Collection

#### 6.2.1 Concepts: Graph, Engine Configs, and Execution Plans

The autotuning preparation pipeline has four stages:

1. **Operation graph**: The computation description (operations, tensor shapes, data types). Built once via `build_operation_graph(handle)`. Not executable at this stage.

2. **Engine configs**: Lightweight descriptors of available engine implementations for the graph. An `EngineConfigInfo` contains the engine ID, name, available knobs, workspace requirements, and whether the engine supports exhaustive benchmarking. No kernels are compiled at this stage.

3. **Plan specs**: Engine configurations with knob settings, collected via `add_engine_*()` calls. These are stored on the graph as specifications (engine ID + knob settings) without compiling any kernels. Plan specs are an intermediate stage between engine configs and compiled plans, representing the set of candidates to be autotuned.

4. **Execution plans**: Compiled, executable kernel plans. Plans are compiled either by `build_plans()` (the existing single-plan non-autotune path) or by `autotune()` (which compiles plan specs into execution plans, then benchmarks them).

The separation between engine configs, plan specs, and execution plans is the key design difference from cuDNN: users can inspect and filter engine configs, create variants, and collect plan specs *before* paying the cost of kernel compilation. Compilation is deferred until `autotune()` is called (see also § A.3 point 3).

#### 6.2.2 Types

```cpp
struct EngineConfigInfo {
    int64_t engineId;
    std::string engineName;
    std::vector<Knob> knobs;           // Available knobs and their constraints
    bool supportsExhaustive;           // Whether this engine has a benchmarking knob
    int64_t workspaceSize;             // Workspace bytes required by this engine
};

struct EngineVariant {
    int64_t engineId;
    std::map<KnobType_t, KnobValueVariant> knobSettings;
};

struct KnobSweepAxis {
    KnobType_t knobId;
    std::vector<KnobValueVariant> values;
};

struct EngineSweepSpec {
    int64_t engineId;
    std::vector<KnobSweepAxis> axes;            // Knobs to sweep (Cartesian product)
    std::map<KnobType_t, KnobValueVariant> fixedSettings;  // Knobs held constant for each combination of axes
};
```

#### 6.2.3 API on Graph

**Engine discovery**:

```cpp
Error get_engine_configs(std::vector<EngineConfigInfo>& configs,
                         const std::vector<HeuristicMode>& modes = {HeuristicMode::FALLBACK});
```

Queries the backend for all engine implementations that *could* handle the current operation graph, ranked by the specified heuristic mode(s). The list may include engines unsupported on the current hardware; these are filtered out during compilation inside `autotune()` (see § 6.2.4).

**Plan spec collection** (for autotuning):

```cpp
// Add plan specs for a set of engine configs using each engine's default knob settings.
// Stores specs without compiling. Compilation is deferred to autotune().
Error add_engine_configs(const std::vector<EngineConfigInfo>& configs);

// Add a single plan spec for one engine, optionally with explicit knob settings.
// Stores the spec without compiling. Compilation is deferred to autotune().
Error add_engine(int64_t engineId,
                 const std::vector<KnobSetting>& knobSettings = {});

// Add plan specs from explicit user-defined variants (engine ID + knob settings).
// Stores specs without compiling. Compilation is deferred to autotune().
Error add_engine_variants(const std::vector<EngineVariant>& variants);

// Add plan specs from a Cartesian product sweep over knob axes for each engine.
// Computes the Cartesian product and stores the resulting specs without compiling.
Error add_engine_sweep(const std::vector<EngineSweepSpec>& specs);
```
All `add_engine_*()` functions store plan specifications (engine ID + knob settings) on the graph without compiling any kernels. Compilation is deferred to `autotune()`. Multiple calls to `add_engine_*()` functions can be made on the same graph (see § 6.2.4).

**Validation**: `add_engine_*()` functions validate knob names and ranges at the time of the call, but do NOT compile. The `global.benchmarking` knob is rejected/stripped — it is managed exclusively by `autotune()` in EXHAUSTIVE mode.

**Convenience shortcut**:

```cpp
// Discover all engines and add plan specs with default knobs. Equivalent to:
//   get_engine_configs(configs);
//   add_engine_configs(configs);
Error add_all_engines();
```

**Workspace query**:

```cpp
// Returns the maximum workspace size across all added plan specs.
// Workspace size is available from EngineConfigInfo::workspaceSize at discovery time.
// Call after add_engine_*(), before allocating workspace for autotune/execute.
Error get_max_workspace_size(int64_t& maxSize) const;
```

**Workspace sizing**:
1. `get_max_workspace_size(maxSize)` populates `maxSize` with the maximum across all added plan specs (using workspace sizes from the engine configs). Use this to allocate workspace when no `maxWorkspaceBytes` cap is set.
2. If `AutotuneConfig.maxWorkspaceBytes` is set, the workspace passed to `autotune()` only needs to be that size, since plans exceeding the cap are skipped.
3. After autotuning, the workspace for `execute()` only needs to accommodate the winner's `workspaceSize` field, which may be smaller.

#### 6.2.4 Plan Spec Collection Semantics

**Additive with deduplication**: All `add_engine_*()` calls append to the same internal plan spec list. Specs are deduplicated by `(engineId, knobSettings)`: if an identical entry already exists, the duplicate is silently skipped. Specs for the same engine with *different* knob settings are distinct entries (the intended use case for variant autotuning).

**Zero specs added**: If no specs were added (the input list was empty), `add_engine_*()` returns an error.

**Empty knob settings = default**: `add_engine(id, {})` is equivalent to `add_engine(id)`; an empty vector means "use engine defaults."

**Benchmarking knob rejection**: The `global.benchmarking` knob is rejected/stripped by all `add_engine_*()` functions. This knob is internal-only, managed exclusively by `autotune()` in EXHAUSTIVE mode.

Multiple `add_engine_*()` calls can be composed freely:

```cpp
// Add default-knob specs for engines you don't want to sweep
graph->add_engine_configs(baseline_configs);

// Add knob-swept variant specs for engines you do want to sweep
graph->add_engine_variants(miopen_variants);

// Dedup ensures no redundant specs even if engine IDs overlap.
// autotune() compiles and benchmarks all specs.
```

**Hardware support filtering**: Hardware compatibility is checked during compilation inside `autotune()`, not during `add_engine_*()`. Unsupported engines (e.g., requiring a hardware feature absent on the current GPU) fail compilation and are excluded from benchmarking. For batch operations, unsupported engines are silently skipped. The `add_engine_*()` functions only validate knob names and ranges.

**Engine ID validation** (checked at add time):
- Nonexistent engine ID (not loaded in hipDNN): hard error.
- Engine exists but doesn't support this graph: silently skipped in batch calls, error for `add_engine()`.
- Nonexistent knob or out-of-range knob value: hard error (message identifies the first invalid entry).
- `global.benchmarking` knob specified: rejected/stripped with a warning.

#### 6.2.5 What Happens Under the Hood

All `add_engine_*()` functions follow the validate-then-store pattern from § 6.2.4. No compilation occurs. Per-function details:

**`add_engine_configs(configs)`**: For each `EngineConfigInfo`, stores a plan spec using the engine's default knob settings. See § 6.2.4 for deduplication behavior.

**`add_engine(engineId, knobSettings)`**, **`add_engine_variants(variants)`**, and **`add_engine_sweep(specs)`** store specs only for the specified engine IDs. If the caller already knows valid engine IDs, these functions can be called directly without `get_engine_configs()`. Discovery is only needed to find available engines or inspect metadata.

**`add_engine_variants(variants)`**: For each `EngineVariant`, stores a plan spec using the specified `engineId` and `knobSettings`. Each variant is equivalent to `add_engine(v.engineId, v.knobSettings)`.

**`add_engine_sweep(specs)`**: For each `EngineSweepSpec`, computes the Cartesian product of all `axes` values, merges each combination with `fixedSettings` to form an `EngineVariant`, then stores each variant as in `add_engine_variants`. For example, 2 axes of 3 values each produces 9 plan specs. Returns an error if the Cartesian product exceeds 10,000 plan specs per call; a warning is logged for products exceeding 1,000.

**Cartesian product example:**

```cpp
EngineSweepSpec spec;
spec.engineId = 42;  // engine name: "MIOPEN_ENGINE_X"
spec.axes = {
    {"SPLIT_K",   {int64_t{1}, int64_t{2}, int64_t{4}}},
    {"TILE_SIZE", {int64_t{0}, int64_t{1}}}
};
spec.fixedSettings = {{"REDUCTION_MODE", int64_t{1}}};
```

The sweep produces 3 x 2 = 6 plan specs:

| Spec | `SPLIT_K` | `TILE_SIZE` | `REDUCTION_MODE` (fixed) |
|------|-----------|-------------|--------------------------|
| 1 | 1 | 0 | 1 |
| 2 | 1 | 1 | 1 |
| 3 | 2 | 0 | 1 |
| 4 | 2 | 1 | 1 |
| 5 | 4 | 0 | 1 |
| 6 | 4 | 1 | 1 |

Each row is stored as a separate plan spec. All six share the fixed `REDUCTION_MODE=1`, while `SPLIT_K` and `TILE_SIZE` vary across all axis value combinations. Compilation into execution plans is deferred to `autotune()`.

#### 6.2.6 Typical Flow

```
1. graph->build_operation_graph(handle)       // build the operation graph
2. graph->get_engine_configs(configs)          // discover engine configs (no compilation)
3. // user inspects, filters, reorders configs
4. graph->add_engine_*(...)                    // validate inputs, store plan specs (no compilation)
5. graph->autotune(...)                        // compile plan specs, then benchmark
```

`add_all_engines()` is a convenience shortcut for steps 2-4 using all engines with default knobs.

> **Note**: The existing non-autotune flow (`build_operation_graph()` → `create_execution_plan_ext()` → `check_support()` → `build_plans()` → `execute()`) remains unchanged. That path produces a single execution plan and is not affected by the autotuning changes.

### 6.3 Autotuning API on Graph

```cpp
Error autotune(
    hipdnn_backend_descriptor_t handle,
    const std::unordered_map<int64_t, void*>& variantPack,
    void* workspace,
    const AutotuneConfig& config = {},
    const AutotuneStorageConfig& storageConfig = {},
    std::vector<AutotuneResult>* results = nullptr);
```

Compiles all stored plan specs into execution plans, benchmarks them, and selects the best. Subsequent `execute()` calls use the winning plan. Pass a non-null `results` pointer to populate it with ranked results; pass `nullptr` or omit the parameter to discard results. If `storageConfig.filePath` is non-empty, results are also written to the config file.

**Compilation inside autotune()**: When `autotune()` is called, it first compiles all stored plan specs into execution plans. In EXHAUSTIVE mode, it first builds temporary priming plans (using each plan spec's knob settings plus `global.benchmarking=1`), executes them once to prime engine caches, discards them, then compiles the real plans (against the now-primed engine state), and finally benchmarks them. In AUTO mode, it compiles plan specs directly into execution plans, then benchmarks.

**Error conditions**: `autotune()` returns an error (with error log) when:
- No plan specs added (empty spec list)
- All plans fail compilation or benchmarking (every result `succeeded = false`)
- HIP memory operations fail
- Workspace pointer null but plans require workspace
- Handle invalid
- `variantPack` missing required tensor pointers
- EXHAUSTIVE mode: priming failure when `continueOnPrimingFailure` is `false` (default)

Failed entries (`succeeded = false`) are always placed after successful entries in the results vector, regardless of ranking. Failed entries have `rank = -1`.

**Plan selection mechanism**: After `autotune()` completes, the winning plan is stored on the graph and subsequent `execute()` calls use it — the same as today's flow after `build(handle)`.

**Ranking function contract**:
- Must only reorder the vector in-place; must not add, remove, or modify elements.
- Receives only succeeded entries; failed entries are appended afterward with `rank = -1`.
- `autotune()` re-assigns `rank` fields (0-based) after the callback returns.
- If the function throws, `autotune()` falls back to default ranking (by `minTimeMs`).

**`engineIdFilter` warnings**: A warning is logged if there are no plan specs for an engine ID listed in `engineIdFilter`.

**Tensor data validity**: Autotuning assumes idempotent execution or that the caller provides separate input/output buffers. Repeated execution with `autotune()` may produce different results for non-idempotent operations (in-place ops, or operations where the output is also an input).

**`HIPDNN_ENGINE_OVERRIDE_FILE` interaction**: Behavior is undefined if the `HIPDNN_ENGINE_OVERRIDE_FILE` env var is set during autotuning.

### 6.4 Benchmarking Flows

**AUTO mode**:
```
1. Compile all plan specs into execution plans
2. For each compiled plan:
     a. device synchronize (ensure GPU is idle before this plan's benchmarking)
     b. warmup iterations (discard timing)
     c. device synchronize
     d. [strategy-specific timed iteration loop]
     e. compute avgTimeMs and track minTimeMs
sort by minTimeMs ascending (or user rankingFn)
```

**EXHAUSTIVE mode**:
```
1. For each plan spec where the engine's supportsExhaustive is true:
     * build a temporary priming plan using the plan spec's knob settings plus global.benchmarking = 1
     * execute the priming plan once
     * discard the priming plan
     * On priming failure:
       - If continueOnPrimingFailure is false (default): abort, return error
       - If continueOnPrimingFailure is true: mark engine as unprimed, continue
2. Compile all plan specs into execution plans (engines are now primed)
3. For each compiled plan:
     a. device synchronize (ensure GPU is idle before this plan's benchmarking)
     b. warmup iterations (discard timing)
     c. device synchronize
     d. [strategy-specific timed iteration loop]
     e. compute avgTimeMs and track minTimeMs
sort by minTimeMs ascending (or user rankingFn)
```

> **Stream**: `autotune()` uses the stream set on the handle via `hipdnnSetStream()`. To benchmark on a specific stream, call `hipdnnSetStream()` before calling `autotune()`.

> **Synchronization**: `hipDeviceSynchronize()` is used before each plan's timed iterations (inside the per-plan loop), preventing cross-plan interference. It is called both before warmup and after warmup (before timed runs). `hipEventSynchronize(stop)` is used between timed iterations (lightweight, per-stream). `hipStreamSynchronize()` is not used during autotuning.

Engines without exhaustive support (where `supportsExhaustive` is false) skip the priming phase (step 1) and are compiled and benchmarked normally.

When `continueOnPrimingFailure` is `true` and an engine's priming fails, the engine is still compiled and benchmarked (against unprimed state). Its `AutotuneResult::ranExhaustive` is set to `false`, and `errorMessage` notes the priming failure reason even though `succeeded` may be `true` (the benchmark itself succeeded).

**Strategy implementations**:

- **SINGLE_SHOT**: One `hipEventRecord` pair around one execution. Fast, rough ranking.
- **FIXED_AVERAGE**: Per-iteration event timing for N executions. Each iteration is independently timed. Reports min, avg, and stddev across all N timings.
- **RUN_UNTIL_STABLE**: Per-iteration event timing. After each iteration, checks if the coefficient of variation of the last `windowSize` timings is below `stabilityThreshold`. Stops when stable or `maxIterations` reached.

**Warmup failure handling**: Plans that fail during warmup are marked `succeeded = false`; the autotuner proceeds to the next plan.

### 6.5 Config File Output

Reuses the `EngineOverrideConfig` JSON format with autotuning metadata added:

```json
{
  "engine_overrides": [
    {
      "op": "conv_fprop",
      "engine_name": "MIOPEN_ENGINE_DETERMINISTIC",
      "tensors": [
        { "dim": [16, 64, 56, 56], "stride": [200704, 3136, 56, 1] },
        { "dim": [64, 64, 3, 3], "stride": [576, 9, 3, 1] }
      ],
      "knobs": [
        { "knob_id": "global.workspace_size_limit", "type": "int", "value": 16777216 },
        { "knob_id": "global.search_mode", "type": "int", "value": 2 }
      ],
      "autotune_metadata": {
        "rank": 0,
        "avg_time_ms": 0.0439,
        "mode": "exhaustive",
        "ran_exhaustive": true,
        "strategy": "run_until_stable",
        "iterations_run": 37,
        "converged": true,
        "device": "gfx942",
        "timestamp": "2026-04-21T10:30:00Z"
      }
    }
  ]
}
```

**Knob settings in config file entries**: Each entry includes a `knobs` array listing the knob ID, type, and value for every non-default knob set when the winning plan was built. This is required to reproduce the autotuned configuration: the engine name alone is insufficient when the winner used non-default knob settings. On load, `EngineOverrideConfig` uses these knob settings to recreate the exact autotuned plan via the existing single-plan build path. If the winner used default knobs, the `knobs` array is empty. The `type` field (`"int"`, `"double"`, or `"string"`) enables correct deserialization, since JSON does not distinguish `int64_t` from `double` for integer values.

When appending, existing rules for the same operation+tensor+knob combination are replaced. By default (`deleteAllExistingFileContent = false`), existing content is preserved (safe append). When `deleteAllExistingFileContent = true`, all existing file content is deleted before writing (destructive, explicit opt-in). The `autotune_metadata` section is informational and does not affect `matchOperation()`, which ignores unknown fields.

**Concurrent access**: Config file append is not safe for concurrent writers. For concurrent scenarios, use separate output files per process and merge afterward.

**Config file match key**: `EngineOverrideConfig::matchOperation()` returns a `MatchResult` struct containing both `engineId` and `knobSettings` (instead of just `int64_t`). This is needed to distinguish between multiple entries with the same (op, tensors) combination that differ only by knob settings, and to recreate the exact autotuned plan via the existing single-plan build path.

**Core operation mapping**: For multi-operation graphs, the config file entry is keyed by the core operation:

1. Convolution, GEMM, SDPA (highest priority)
2. Normalization
3. Pointwise (lowest priority)

### 6.6 Checkpoint/Resume

When `checkpointFile` is set in `AutotuneConfig`:
1. Before starting, check if the file exists
2. If it exists: load partial results, validate session ID (`hash(graphSig + devSig + configHash + workspaceSize + planSpecListHash)`), skip completed plan specs
3. After each plan spec: atomically write updated results (rename-on-write)
4. After all plan specs complete: write final results

Stale checkpoints (session ID mismatch, including changed plan specs) are discarded.

The hash algorithm used for converting engine names to IDs can be used for the graph signature. The hash must be deterministic and strict: the checkpoint should be ignored if anything changes on the system (GPU architecture, driver version, library version). The hash must be consistent per machine, per graph, and per engine config. The hash provides some safety
against resuming the autotune in a different environment, but it is the user's responsibility
to ensure consistent environments when resuming autotune runs.

---

## 7. Porting Guide: cuDNN → hipDNN

For API mapping, key differences, and complete porting examples, see Appendix A. For concrete use cases, see Appendix B.

---

## 8. Complete Example

```cpp
#include <hipdnn_frontend.h>

int main() {
    hipdnnHandle_t handle;
    hipdnnCreate(&handle);

    auto graph = std::make_shared<hipdnn_frontend::Graph>();
    // ... configure graph ...
    graph->validate();
    graph->build_operation_graph(handle);

    // Discover and filter engines
    std::vector<hipdnn_frontend::EngineConfigInfo> configs;
    graph->get_engine_configs(configs);

    std::erase_if(configs, [](const auto& c) { return c.workspaceSize > (256 << 20); });

    // Step 1: Add default-knob plan specs per engine config (up to N specs).
    graph->add_engine_configs(configs);

    // Step 2: Add knob-swept plan specs for MIOpen engines (3 variants each).
    std::vector<hipdnn_frontend::EngineVariant> variants;
    for (const auto& config : configs) {
        if (config.engineName.find("MIOPEN") != std::string::npos) {
            for (int64_t ws : {int64_t{0}, int64_t{1 << 20}, int64_t{16 << 20}}) {
                variants.push_back({
                    .engineId = config.engineId,
                    .knobSettings = {{"global.workspace_size_limit", ws}}
                });
            }
        }
    }
    // Duplicates from Step 1 are skipped in add_engine_variants().
    graph->add_engine_variants(variants);

    int64_t maxWs;
    graph->get_max_workspace_size(maxWs);
    void* workspace;
    hipMalloc(&workspace, maxWs);

    std::unordered_map<int64_t, void*> variantPack;
    // ... populate ...

    // Autotune with convergence-based strategy, persist to config file
    std::vector<hipdnn_frontend::AutotuneResult> results;
    auto err = graph->autotune(handle, variantPack, workspace,
        {.mode = hipdnn_frontend::TuneMode::EXHAUSTIVE,
         .strategy = hipdnn_frontend::AutotuneStrategy::RUN_UNTIL_STABLE,
         .maxIterations = 50,
         .stabilityThreshold = 0.03f,
         .checkpointFile = "autotune.ckpt"},
        {.filePath = "autotune_results.json"},
        &results);

    // Execute with autotuned winner
    graph->execute(handle, variantPack, workspace);

    hipFree(workspace);
    hipdnnDestroy(handle);
    return 0;
}
```

---

## 9. Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| **Per-iteration event overhead** | SINGLE_SHOT avoids per-iteration event overhead for sub-microsecond kernels. FIXED_AVERAGE provides deterministic iteration count (predictable runtime), while RUN_UNTIL_STABLE provides adaptive iteration count (potentially faster for engines that converge quickly). |
| **Cartesian product growth in knob sweeps** | Configurable upper bound (error at 10,000 plan specs per `add_engine_sweep()` call, warning at 1,000). User controls axes and values per `EngineSweepSpec`; validate-then-store rejects invalid knobs early. |
| **Config file format changes** | Backward-compatible: `matchOperation()` ignores unknown fields. |
| **Config file conflicts on append** | Existing rules for the same (op, tensors, knobs) are replaced, preventing stale entries. |
| **Checkpoint session validation** | Discard stale checkpoints (session ID mismatch, including plan spec changes → start fresh). |

---

## Appendix A: Porting Guide from cuDNN Autotuning to hipDNN

This appendix covers how hipDNN autotuning differs from cuDNN and what must change when porting.

For the cuDNN sample this guide references, see [cudnn-frontend/samples/cpp/misc/autotuning.cpp](https://github.com/NVIDIA/cudnn-frontend/blob/f26b794e2d36f40cf14fd4af8919d2b097dc546f/samples/cpp/misc/autotuning.cpp#L4).

### A.1 Auto Workflow Comparison Examples

These examples illustrate the similarities and differences between hipDNN and cuDNN autotuning. See A.3 for detailed descriptions.

#### API Equivalence

hipDNN has direct equivalents for most cuDNN autotuning functions.

<table>
<tr><th>cuDNN</th><th>hipDNN</th></tr>
<tr>
<td><pre lang="cpp">
graph.validate();
graph.build_operation_graph(handle);
graph.create_execution_plans({HeurMode_t::A});
graph.check_support();
graph.build_plans(BuildPlanPolicy_t::ALL);

int64_t ws;
ws = graph.get_autotune_workspace_size();
void* workspace;
cudaMalloc(&workspace, ws);

std::unordered_map<int64_t, void*> variant_pack =
    {{a_uid, a_ptr}, {w_uid, w_ptr}, {y_uid, y_ptr}};

graph.autotune(handle, variant_pack, workspace);

graph.execute(handle, variant_pack, workspace);
cudaFree(workspace);
</pre></td>
<td><pre lang="cpp">
graph.validate();
graph.build_operation_graph(handle);
graph.add_all_engines();
//
//

int64_t ws;
graph.get_max_workspace_size(ws);
void* workspace;
hipMalloc(&workspace, ws);

std::unordered_map<int64_t, void*> variant_pack =
    {{a_uid, a_ptr}, {w_uid, w_ptr}, {y_uid, y_ptr}};

graph.autotune(handle, variant_pack, workspace);

graph.execute(handle, variant_pack, workspace);
hipFree(workspace);
</pre></td>
</tr>
</table>

#### Simple Common Autotune Workflow With hipDNN Extensions

cuDNN's `autotune()` benchmarks internally and reorders plans, returning `error_t`. hipDNN's `autotune()` returns `Error` and optionally populates a `results` pointer with ranked results (pass `nullptr` or omit to discard). It can also persist results to a config file.

<table>
<tr><th>cuDNN</th><th>hipDNN</th></tr>
<tr>
<td><pre lang="cpp">
graph.validate();
graph.build_operation_graph(handle);
graph.create_execution_plans({HeurMode_t::A});
graph.check_support();
graph.build_plans(BuildPlanPolicy_t::ALL);
void* workspace =
    allocate(graph.get_autotune_workspace_size());
graph.autotune(handle, variant_pack, workspace);
//
//
graph.execute(handle, variant_pack, workspace);
</pre></td>
<td><pre lang="cpp">
graph.validate();
graph.build_operation_graph(handle);
//
//
graph.add_all_engines();
int64_t ws;
graph.get_max_workspace_size(ws);
void* workspace = allocate(ws);
graph.autotune(handle, variant_pack, workspace,
    {.mode = TuneMode::EXHAUSTIVE},
    {.filePath = "autotune_results.json"});
graph.execute(handle, variant_pack, workspace);
</pre></td>
</tr>
</table>

#### With Engine Filtering and hipDNN Autotune Extensions

Engine configurations can be filtered directly before adding plan specs.

<table>
<tr><th>cuDNN</th><th>hipDNN</th></tr>
<tr>
<td><pre lang="cpp">
graph.validate();
graph.build_operation_graph(handle);

graph.create_execution_plans({HeurMode_t::A});
graph.check_support();
// Opaque filtering (no engine inspection):
graph.deselect_workspace_greater_than(256 << 20);
graph.deselect_engines({"engine_1", "engine_2"});
//
//
graph.build_plans(BuildPlanPolicy_t::ALL);

void* workspace =
    allocate(graph.get_autotune_workspace_size());
graph.autotune(handle, variant_pack, workspace);
//
//
//
graph.execute(handle, variant_pack, workspace);
</pre></td>
<td><pre lang="cpp">
graph.validate();
graph.build_operation_graph(handle);

std::vector<EngineConfigInfo> configs;
graph.get_engine_configs(configs);
std::erase_if(configs, [](const auto& c) {
    return c.workspaceSize > (256 << 20)
        || c.engineName == "engine_1"
        || c.engineName == "engine_2";
});
graph.add_engine_configs(configs);

int64_t ws;
graph.get_max_workspace_size(ws);
void* workspace = allocate(ws);
graph.autotune(handle, variant_pack,
    workspace,
    {.mode = TuneMode::EXHAUSTIVE},
    {.filePath = "autotune_results.json"});
graph.execute(handle, variant_pack, workspace);
</pre></td>
</tr>
</table>

#### cuDNN Manual Benchmarking Loop vs. hipDNN `autotune()`

The [cuDNN autotuning sample](https://github.com/NVIDIA/cudnn-frontend/blob/f26b794e2d36f40cf14fd4af8919d2b097dc546f/samples/cpp/misc/autotuning.cpp#L4) implements a manual benchmarking loop using `execute_plan_at_index()`. hipDNN's `autotune()` replaces this entire pattern.

<table>
<tr><th>cuDNN (manual benchmarking loop)</th><th>hipDNN (RFC)</th></tr>
<tr>
<td><pre lang="cpp">
graph.validate();
graph.build_operation_graph(handle);
graph.create_execution_plans({HeurMode_t::A});
graph.check_support();
graph.build_plans(BuildPlanPolicy_t::ALL);

auto plan_count =
    graph.get_execution_plan_count();

// Find max workspace across all plans
int64_t ws = 0;
for (int i = 0; i < plan_count; i++)
    ws = std::max(ws,
        graph.get_workspace_size_plan_at_index(i));
void* workspace;
cudaMalloc(&workspace, ws);

// Benchmark each plan
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);
std::vector<float> times(plan_count, 10.0f);

for (int i = 0; i < plan_count; i++) {
    // Warmup
    auto err = graph.execute_plan_at_index(
        handle, variant_pack, workspace, i);
    if (err.is_bad()) continue;
    cudaDeviceSynchronize();

    // Timed iterations
    cudaEventRecord(start, stream);
    for (int iter = 0; iter < 10; iter++)
        graph.execute_plan_at_index(
            handle, variant_pack, workspace, i);
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    times[i] = ms / 10.0f;
}

// Select winner
auto best = std::min_element(
    times.begin(), times.end());
auto idx = std::distance(times.begin(), best);

graph.build_plan_at_index(idx);
graph.execute(handle, variant_pack, workspace);
</pre></td>
<td><pre lang="cpp">
graph.validate();
graph.build_operation_graph(handle);
//
//
graph.add_all_engines();
//
//
//
//
//
//
//
int64_t maxWs;
graph.get_max_workspace_size(maxWs);
void* workspace;
hipMalloc(&workspace, maxWs);

// autotune() compiles + benchmarks all plan specs
std::vector<AutotuneResult> results;
graph.autotune(handle,
    variant_pack, workspace,
    {.mode = TuneMode::EXHAUSTIVE,
     .warmupIterations = 1,
     .timedIterations = 10},
    {}, &results);
printf("Winner: %s (%.4f ms)\n",
    results[0].engineName.c_str(),
    results[0].minTimeMs);
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
graph.execute(handle, variant_pack, workspace);
</pre></td>
</tr>
</table>

hipDNN's `autotune()` replaces the ~40-line cuDNN manual benchmarking loop with a single configurable call.


### A.2 Step-by-Step API Mapping

Status tags: **existing** = already in hipDNN, unchanged; **RFC** = new, proposed by this RFC.

| cuDNN | hipDNN | Source | Notes |
|-------|--------|--------|-------|
| `graph.validate()` | `graph.validate()` | **existing** | Identical |
| `graph.build_operation_graph(handle)` | `graph.build_operation_graph(handle)` | **existing** | Identical |
| `graph.create_execution_plans({HeurMode_t::A})` | `graph.get_engine_configs(configs)` | **RFC** | cuDNN discovers engines internally; hipDNN exposes engine list for inspection |
| `graph.check_support()` | _(handled during compilation inside `autotune()`)_ | **RFC** | hipDNN filters unsupported engines during compilation inside `autotune()` |
| `graph.build_plans(BuildPlanPolicy_t::ALL)` | `graph.add_engine_configs(configs)` | **RFC** | Or use `add_all_engines()` (**RFC**) as a convenience shortcut that skips inspection |
| `graph.deselect_workspace_greater_than(n)` | Filter `configs` before `add_engine_configs()` | **RFC** | User code pattern; no dedicated API (see example below) |
| `graph.deselect_engines(barred)` | Filter `configs` by `engineName` | **RFC** | User code pattern; no dedicated API (see example below) |
| `graph.get_autotune_workspace_size()` | `graph.get_max_workspace_size(maxSize)` | **RFC** | Same concept; hipDNN uses out-parameter pattern |
| `graph.autotune(handle, variant_pack, workspace)` | `graph.autotune(handle, variant_pack, workspace, config)` | **RFC** | Compiles plan specs, benchmarks, and selects winner; returns `Error`. Optional trailing `results` pointer for ranked results |
| `graph.execute(handle, variant_pack, workspace)` | `graph.execute(handle, variant_pack, workspace)` | **existing** | Identical |
| `graph.get_execution_plan_count()` | _(no equivalent; managed internally by `autotune()`)_ | **RFC** | Multi-plan support exists exclusively for autotuning |
| `graph.execute_plan_at_index(handle, ...)` | _(no equivalent; `autotune()` handles benchmarking)_ | **RFC** | Manual benchmarking loops removed; use `autotune()` |
| `graph.build_plan_at_index(idx)` | _(no equivalent; `autotune()` selects winner)_ | **RFC** | Manual plan selection removed; `autotune()` selects the winner |

> **Note**: The existing `create_execution_plans()` / `check_support()` / `build_plans()` path continues to work for simple non-autotune cases. The new `get_engine_configs()` / `add_engine_configs()` path provides engine inspection, filtering, and plan spec collection for autotuning. Both paths are valid.

### A.3 Key Differences and Similarities

**1. Tensor pointer mapping (identical)**

Both cuDNN and hipDNN use `std::unordered_map<int64_t, void*>` directly (tensor UID → device pointer) with no wrapper type. The code is identical:

```cpp
// cuDNN and hipDNN - same type, same usage:
std::unordered_map<int64_t, void*> variant_pack = {
    {x_uid, x_ptr}, {w_uid, w_ptr}, {y_uid, y_ptr}};
```

**2. `autotune()` return semantics**

Both select the winner and mutate graph state so `execute()` uses the autotuned plan. cuDNN returns `error_t` (no results). hipDNN returns `Error` and accepts an optional trailing `results` pointer. Pass `nullptr` or omit for cuDNN-compatible behavior (no results). Pass a non-null pointer to populate it with ranked results:

```cpp
// cuDNN - no access to results:
graph.autotune(handle, variant_pack, workspace);

// hipDNN - cuDNN-compatible, no results (results pointer omitted):
auto err = graph.autotune(handle, variant_pack, workspace, config);

// hipDNN - same behavior, plus access to all results:
std::vector<AutotuneResult> results;
auto err = graph.autotune(handle, variant_pack, workspace, config, storageConfig, &results);
for (const auto& r : results)
    printf("#%d %-30s min=%.4f ms\n", r.rank, r.engineName.c_str(), r.minTimeMs);
// Winner is results[0]; graph is configured so execute() uses that plan.
```

**3. Engine discovery is collapsed in cuDNN but separated in hipDNN**

cuDNN uses `create_execution_plans()` + `check_support()` as opaque steps. hipDNN's `get_engine_configs()` returns a user-inspectable list. For simple ports, `add_all_engines()` gives the same one-call behavior:

```cpp
// cuDNN (3 calls, no inspection):
graph.create_execution_plans({HeurMode_t::A});
graph.check_support();
graph.build_plans(BuildPlanPolicy_t::ALL);

// hipDNN - simple path (1 call, no inspection):
graph.add_all_engines();

// hipDNN - with inspection (2 calls):
std::vector<EngineConfigInfo> configs;
graph.get_engine_configs(configs);
// ... inspect, filter, reorder ...
graph.add_engine_configs(configs);
```

**4. Benchmarking parameters are configurable**

cuDNN hardcodes 100 max iterations and a 0.95 convergence threshold. hipDNN exposes these via `AutotuneConfig`. Defaults produce reasonable behavior, so a minimal port needs no configuration:

```cpp
// cuDNN - hardcoded parameters:
graph.autotune(handle, variant_pack, workspace);

// hipDNN - equivalent default behavior (same 3 arguments):
graph.autotune(handle, variant_pack, workspace);

// hipDNN - with explicit configuration:
graph.autotune(handle, variant_pack, workspace,
    {.mode = TuneMode::EXHAUSTIVE,
     .strategy = AutotuneStrategy::RUN_UNTIL_STABLE,
     .maxIterations = 100,
     .stabilityThreshold = 0.02f});
```

**5. Plan filtering uses get-filter-add instead of dedicated filter methods**

cuDNN provides specific filter methods (`deselect_workspace_greater_than`, `deselect_engines`). hipDNN uses the general-purpose `get_engine_configs()` → filter → `add_engine_configs()` pattern. See the "With Engine Filtering" example in § A.1.

**6. hipDNN extended autotune features**

| hipDNN Feature | cuDNN Equivalent |
|----------------|------------------|
| `AutotuneResult` vector with per-engine timing data | None (cuDNN autotune is opaque) |
| `TuneMode::EXHAUSTIVE` (benchmarking knob priming) | None (cuDNN has no equivalent knob) |
| `AutotuneStrategy` (SINGLE_SHOT, FIXED_AVERAGE, RUN_UNTIL_STABLE) | Hardcoded convergence strategy |
| `AutotuneConfig` (warmup, iterations, workspace limit, priming failure control) | Hardcoded defaults |
| Config file output (`AutotuneStorageConfig`) | None |
| Knob variant autotuning (`EngineVariant`, `EngineSweepSpec`, `add_engine_*()`) | Limited `create_execution_plan(id, knob_map)` |
| Custom ranking function (`AutotuneRankingFn`) | None |
| Checkpoint/resume | None |

---

## Appendix B: Use Cases

Concrete use cases with code examples.

All examples assume standard setup boilerplate:
```cpp
hipdnnHandle_t handle;
hipdnnCreate(&handle);

auto graph = std::make_shared<hipdnn_frontend::Graph>();
// ... configure graph operations, tensors, etc. ...
graph->validate();
graph->build_operation_graph(handle);
```

### Core Autotuning

**1. Quick autotune.** Add all engine specs, benchmark with simple wall-time, execute the winner.

```cpp
graph->add_all_engines();
int64_t maxWs;
graph->get_max_workspace_size(maxWs);
void* workspace = allocate(maxWs);

graph->autotune(handle, variantPack, workspace,
    {.mode = TuneMode::AUTO});
graph->execute(handle, variantPack, workspace);
```

**2. Exhaustive autotune.** Primes engine-internal caches (e.g., MIOpen's `find` phase) via temporary priming plans before timing; more accurate for engines with lazy compilation.

```cpp
graph->add_all_engines();
int64_t maxWs;
graph->get_max_workspace_size(maxWs);
void* workspace = allocate(maxWs);

graph->autotune(handle, variantPack, workspace,
    {.mode = TuneMode::EXHAUSTIVE});
graph->execute(handle, variantPack, workspace);
```

**3. Inspect autotune results.** Get ranked results for logging or programmatic decisions.

```cpp
graph->add_all_engines();
int64_t maxWs;
graph->get_max_workspace_size(maxWs);
void* workspace = allocate(maxWs);

std::vector<AutotuneResult> results;
graph->autotune(handle, variantPack, workspace,
    {.mode = TuneMode::EXHAUSTIVE},
    {}, &results);

for (const auto& r : results) {
    printf("#%d %-30s min=%.4f ms  avg=%.4f ms  ws=%lld %s\n",
           r.rank, r.engineName.c_str(), r.minTimeMs, r.avgTimeMs,
           r.workspaceSize, r.succeeded ? "OK" : r.errorMessage.c_str());
}
```

**4. Custom warmup/iteration counts.** Override defaults for workloads that need more (or fewer) iterations.

```cpp
graph->autotune(handle, variantPack, workspace,
    {.mode = TuneMode::EXHAUSTIVE,
     .warmupIterations = 10,
     .timedIterations = 50});
```

**5. Custom stream.** Run benchmarks on a user-provided HIP stream. Set the stream on the handle before calling `autotune()`.

```cpp
hipStream_t myStream;
hipStreamCreate(&myStream);
hipdnnSetStream(handle, myStream);

graph->autotune(handle, variantPack, workspace,
    {.mode = TuneMode::AUTO});
```

**6. Workspace-constrained autotuning.** Skip engines that need more workspace than available.

```cpp
graph->autotune(handle, variantPack, workspace,
    {.mode = TuneMode::AUTO,
     .maxWorkspaceBytes = 64 << 20});  // 64 MB limit
```

**7. Autotune specific engines only.** Benchmark only a subset by engine ID.

```cpp
graph->autotune(handle, variantPack, workspace,
    {.mode = TuneMode::EXHAUSTIVE,
     .engineIdFilter = {42, 17}});
```

**8. Programmatic configuration.** Configure mode, warmup, iterations, and output file programmatically.

```cpp
graph->autotune(handle, variantPack, workspace,
    {.mode = TuneMode::EXHAUSTIVE,
     .warmupIterations = 5,
     .timedIterations = 20},
    {.filePath = "autotune_results.json"});
```

### Engine Discovery and Filtering

**9. Inspect engines before adding plan specs.** Retrieve engine configs, inspect metadata, then add plan specs.

```cpp
std::vector<EngineConfigInfo> configs;
graph->get_engine_configs(configs);

printf("Found %zu engines:\n", configs.size());
for (const auto& c : configs)
    printf("  [%lld] %s  exhaustive=%s  knobs=%zu\n",
           c.engineId, c.engineName.c_str(),
           c.supportsExhaustive ? "yes" : "no", c.knobs.size());

graph->add_engine_configs(configs);
```

**10. Exclude specific engines by name.** Remove engines before adding plan specs.

```cpp
std::vector<EngineConfigInfo> configs;
graph->get_engine_configs(configs);

configs.erase(std::remove_if(configs.begin(), configs.end(),
    [](const auto& c) {
        return c.engineName.find("DETERMINISTIC") != std::string::npos;
    }),
    configs.end());

graph->add_engine_configs(configs);
```

**11. Benchmark only exhaustive-capable engines.** Filter to engines supporting benchmarking knob priming.

```cpp
std::vector<EngineConfigInfo> configs;
graph->get_engine_configs(configs);

configs.erase(std::remove_if(configs.begin(), configs.end(),
    [](const auto& c) { return !c.supportsExhaustive; }),
    configs.end());

graph->add_engine_configs(configs);
graph->autotune(handle, variantPack, workspace,
    {.mode = TuneMode::EXHAUSTIVE});
```

**12. Add a single engine for targeted autotuning or debugging.**

```cpp
graph->add_engine(42, {{"global.workspace_size_limit", int64_t{16 << 20}}});
graph->autotune(handle, variantPack, workspace);
graph->execute(handle, variantPack, workspace);
```

### Benchmarking Strategies

**13. Single-shot benchmarking.** One timed iteration per plan for a fast, rough ranking.

```cpp
graph->autotune(handle, variantPack, workspace,
    {.mode = TuneMode::AUTO,
     .strategy = AutotuneStrategy::SINGLE_SHOT});
```

**14. Convergence-based benchmarking.** Run until timing stabilizes.

```cpp
graph->autotune(handle, variantPack, workspace,
    {.mode = TuneMode::EXHAUSTIVE,
     .strategy = AutotuneStrategy::RUN_UNTIL_STABLE,
     .maxIterations = 100,
     .windowSize = 5,
     .stabilityThreshold = 0.02f});
```

### Knob Variant Autotuning

**15. Benchmark an engine with different knob settings.** Compare workspace size trade-offs.

```cpp
std::vector<EngineConfigInfo> configs;
graph->get_engine_configs(configs);

std::vector<EngineVariant> variants;
for (const auto& config : configs) {
    if (config.engineName.find("MIOPEN") != std::string::npos) {
        for (int64_t ws : {int64_t{0}, int64_t{1 << 20}, int64_t{16 << 20}, int64_t{256 << 20}}) {
            variants.push_back({
                .engineId = config.engineId,
                .knobSettings = {{"global.workspace_size_limit", ws}}
            });
        }
    } else {
        variants.push_back({.engineId = config.engineId, .knobSettings = {}});
    }
}

graph->add_engine_variants(variants);
std::vector<AutotuneResult> results;
graph->autotune(handle, variantPack, workspace,
    {.mode = TuneMode::EXHAUSTIVE},
    {}, &results);

for (const auto& r : results)
    printf("#%d %-25s min=%.4f ms\n",
           r.rank, r.engineName.c_str(), r.minTimeMs);
```

**16. Automated knob sweep.** Specify axes; the framework generates the Cartesian product.

```cpp
std::vector<EngineConfigInfo> configs;
graph->get_engine_configs(configs);

auto miopenId = configs[0].engineId;  // assuming first is MIOpen

EngineSweepSpec spec;
spec.engineId = miopenId;
spec.axes = {
    {"global.workspace_size_limit",
     {int64_t{0}, int64_t{1 << 20}, int64_t{16 << 20}, int64_t{256 << 20}}}
};
// Generates 4 variants with workspace_size_limit values: 0, 1M, 16M, 256M

graph->add_engine_sweep({spec});
graph->autotune(handle, variantPack, workspace,
    {.mode = TuneMode::EXHAUSTIVE});
```

### Custom Ranking

**17. Custom ranking.** Rank by workspace size (smallest first), breaking ties by speed.

```cpp
graph->autotune(handle, variantPack, workspace,
    {.mode = TuneMode::EXHAUSTIVE,
     .rankingFn = [](std::vector<AutotuneResult>& results) {
         std::sort(results.begin(), results.end(), [](const auto& a, const auto& b) {
             if (a.workspaceSize != b.workspaceSize)
                 return a.workspaceSize < b.workspaceSize;
             return a.minTimeMs < b.minTimeMs;
         });
     }});
```

### Config File Output and Reuse

**18. Save results to config file.** Write in `EngineOverrideConfig` format for reuse via `HIPDNN_ENGINE_OVERRIDE_FILE`.

```cpp
graph->autotune(handle, variantPack, workspace,
    {.mode = TuneMode::EXHAUSTIVE},
    {.filePath = "autotune_results.json"});
```

**19. Overwrite config file.** Delete existing file content and write fresh results.

```cpp
graph->autotune(handle, variantPack, workspace,
    {.mode = TuneMode::EXHAUSTIVE},
    {.filePath = "my_engine_overrides.json",
     .deleteAllExistingFileContent = true});
```

**20. Reuse autotuned results.** Load a saved config file to select the autotuned engine without re-benchmarking.

```bash
# First run: autotune and save
./my_app  # writes autotune_results.json

# Subsequent runs: skip autotuning, use saved results
export HIPDNN_ENGINE_OVERRIDE_FILE=autotune_results.json
./my_app  # EngineOverrideConfig picks the saved winner
```

**21. Build a library of autotuned configurations.** Autotune multiple graphs across runs, accumulating results into one config file.

```bash
# Run 1: autotune convolution graphs
./tune_conv_graphs --output my_overrides.json

# Run 2: autotune matmul graphs, results are appended by default
./tune_matmul_graphs --output my_overrides.json

# Production: use the combined file
export HIPDNN_ENGINE_OVERRIDE_FILE=my_overrides.json
./my_training_app
```

### Checkpoint/Resume

**22. Resume interrupted autotuning.** Checkpoint file is atomically written after each plan spec. On restart, completed plan specs are skipped. Stale checkpoints (changed graph, device, config, or plan specs) are automatically discarded (see § 6.6).

```cpp
graph->autotune(handle, variantPack, workspace,
    {.mode = TuneMode::EXHAUSTIVE,
     .checkpointFile = "autotune.ckpt"},
    {.filePath = "autotune_results.json"});
// If interrupted and restarted, the same call resumes from where it left off.
```

**23. CI/CD time-budgeted autotuning.** Over multiple CI runs, all plan specs eventually get benchmarked.

```cpp
// CI job (may time out):
graph->autotune(handle, variantPack, workspace,
    {.mode = TuneMode::EXHAUSTIVE,
     .checkpointFile = "/ci/artifacts/autotune.ckpt"},
    {.filePath = "/ci/artifacts/autotune_results.json"});
// Next CI run picks up where the previous one left off.
```
