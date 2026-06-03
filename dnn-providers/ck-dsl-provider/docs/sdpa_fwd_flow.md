# hipDNN SDPA-forward: Graph → kernel execution flow

End-to-end data flow for a scaled-dot-product-attention forward op, from a
hipDNN frontend graph down to GPU kernel launch on the ck-dsl-provider's
unified tiled-2D attention kernel (op-kind `sdpa_fmha_fwd_unified`).

File:line anchors are approximate (they drift with edits); use them as
starting points, not exact addresses.

## Layers crossed

```
 Frontend (C++ graph API)         graph::Graph / SdpaFwdNode / SdpaAttributes
   │  build(handle)
 Backend (C-API descriptors)      HIPDNN_BACKEND_OPERATION_SDPA_FWD_DESCRIPTOR
   │  serialize → FlatBuffer (sdpa_attributes.fbs)
 Plugin C-ABI boundary (dlopen)   hipdnnEnginePlugin* symbols ── EngineManager
   │
 Provider engine (C++)            CkDslSdpaEngine → SdpaFwdPlanBuilder
   │  SdpaAdapter::buildSpec
 Provider spec + dispatcher (C++) SdpaSpec → SdpaSelectionProblem → selectPerfKnobs
   │  sdpaSpecToPayload → CompileServiceBridge (pybind embed)
 Python compile service           compile_service.py → _tiled_2d_impl(arch)
   │
 CK DSL (Python codegen)          build_unified_attention_2d_tiled → compile_kernel → HSACO
   │
 Runtime (C++)                    JitCache → HipModule → SdpaFwdPlan::execute → launch
```

There are **two distinct "specs"**, bridged by a payload dict:

- `SdpaSpec` — provider-side C++, built from the graph.
- `UnifiedAttention2DTiledSpec` — the CK DSL kernel spec (Python).

---

## 1. Frontend: build the graph

- `graph->sdpa(q, k, v, SdpaAttributes)` allocates the output tensor and emplaces
  an `SdpaFwdNode` — `frontend/.../Graph.hpp:2884`. The node holds a
  `graph::SdpaAttributes` (`frontend/.../attributes/SdpaAttributes.hpp:71`)
  carrying every knob (causal, scale, left/right bound, paged via
  `Page_table_K/V`, sinks). B/H/S/D come from the tensor shapes.
- `graph->build(handle)` (`Graph.hpp:1733`) runs validate → infer (output shape
  `{B, Hq, Sq, Dv}`) → `create_execution_plans` → `check_support` →
  `build_plans`.

## 2. Frontend → backend → FlatBuffer

- `createSdpaFwdOperation` packs the node into a backend C-API descriptor
  `HIPDNN_BACKEND_OPERATION_SDPA_FWD_DESCRIPTOR` —
  `frontend/.../detail/SdpaFwdPacker.hpp:17`.
- The backend serializes that descriptor into the **FlatBuffer** `SdpaAttributes`
  (schema `flatbuffers_sdk/schemas/sdpa_attributes.fbs`) —
  `backend/src/descriptors/SdpaFwdOperationDescriptor.cpp:816`.
- **This FlatBuffer op-graph is what crosses to the provider** — the
  frontend graph objects never reach the plugin.

## 3. Plugin boundary: engine discovery + selection

- The backend `dlopen`s the provider and `dlsym`s the standard engine-plugin
  C-ABI (`hipdnnEnginePluginGetApplicableEngineIds`, `…CreateExecutionContext`,
  `…ExecuteOpGraph`, …) — `backend/src/plugin/EnginePlugin.cpp:48-92`.
- Those exports are **auto-generated** from the plugin SDK: the provider sets a
  few macros (`HIPDNN_PLUGIN_CONTAINER_TYPE = CkDslContainer`, …) and
  `#include <hipdnn_plugin_sdk/EnginePluginImpl.inl>` —
  `src/CkDslPluginPublic.cpp:14-21`. The `.inl` wraps the FlatBuffer as an
  `IGraph` and calls into `EngineManager`.
- The provider registers its engine with a hashed string id:
  `HIPDNN_REGISTER_ENGINE(CK_DSL_SDPA_ENGINE, "ck_dsl_sdpa_engine")` → factory
  builds a `CkDslSdpaEngine` — `src/CkDslContainer.cpp:29,55`. The backend asks
  each engine `getApplicableEngineIds` and selects among the applicable ones.

## 4. Engine → plan-builder: the applicability gate (`isApplicable`)

- `CkDslSdpaEngine` owns an `SdpaFwdPlanBuilder` and routes
  `isApplicable`/`buildPlan` to it — `src/engines/sdpa/CkDslSdpaEngine.cpp:29,82`.
  The object passed in is the FlatBuffer **`IGraph& opGraph`**; tensors come via
  `opGraph.getTensorMap()`.
- `SdpaFwdPlanBuilder::isApplicable` (`src/engines/sdpa/SdpaFwdPlanBuilder.cpp:147`)
  does a *trial* `buildSpec`, detects the device arch, then **delegates the real
  capability verdict to Python**: `sdpaSpecToPayload` →
  `CompileServiceBridge::isApplicable` → `compile_service.is_applicable(...)` →
  the DSL gate `supports_tiled_2d(arch)` (where gfx942 is admitted via
  `_tiled_2d_impl`). No kernel is built here — it is the cheap gate.

## 5. Graph → `SdpaSpec` (the graph→spec mapping)

`SdpaAdapter::buildSpec(sdpaAttr, tensorMap)` — `src/adapters/sdpa/SdpaAdapter.cpp:195`
— is the core "map a graph into the spec" step. It:

- reads Q/K/V/O tensors by UID; derives **B/Hq/Sq/D** from Q dims `[B,H,S,D]`;
  validates GQA divisibility, head_size ∈ {64,128,256}, seqlens % 16;
- **causal decision** (`:401-407`): `causal = causal_mask || left_bound > 0`;
  this provider is **causal-only** — a non-causal request is *declined* here;
  `mask_mode = "causal"`; `sliding_window = left_bound`;
- **scale decision** (`:560-564`): `attn_scale_value` or `1/sqrt(D)`, folded to
  **log2 space** (`scale_log2 = scale * log2(e)`) — a launch-time value;
- **paged-vs-dense** (`:474-521`): paged iff `Page_table_K/V` present
  (→ `is_paged`, `block_size = derivePagedBlockSize`); dense leaves
  `block_size = 0` (finalized later in `buildPlan`);
- **varlen** (`:461`) iff both seqlen tensors present; **sinks** (`:525`) iff
  `sink_token` present; extracts strides.
- → populates `SdpaSpec` (`src/adapters/sdpa/SdpaSpec.hpp:72`, nested
  `SdpaProblem` at `:43`).

## 6. Spec → dispatcher (knob selection): `buildPlan` commits

`SdpaFwdPlanBuilder::buildPlan` (`SdpaFwdPlanBuilder.cpp:226`) is where the op is
committed:

1. real `buildSpec`; `detectDeviceArch`; finalize the dense `block_size`
   (`chooseDegenerateBlockSize`, `:264`).
2. **`SdpaSelectionProblem`** built from spec + arch
   (`buildSelectionProblem(spec, arch)`, `:271`).
3. **`enumerateCandidates(selProblem)`** (`:275`) — the buildable knob configs,
   gated per-arch by the LDS budget (gfx942 64 KB / gfx950 160 KB) in
   `supportsTiled2d`.
4. **`spec.knobs = selectPerfKnobs(selProblem, candidates, scorer)`** (`:299`)
   — *the dispatcher*. On gfx950 it runs the LightGBM ML heuristic
   (`selectArgmax` over `scorer.predict`); on **gfx942 it forces
   `selectAnalyticFallback`** (the model is gfx950-trained) —
   `src/adapters/sdpa/SdpaCandidateSelector.cpp:495`. The 9 chosen knobs
   (num_warps, block_m_per_warp, tile_size, mfma32, trqk, regpv, earlyv, …) are
   baked into the spec.

The dispatcher = `SdpaSelectionProblem` → `enumerateCandidates` →
`selectPerfKnobs`. Its output (the knobs) is both a **codegen input** and part of
the **cache key**.

## 7. Spec → payload → DSL kernel build → compile (codegen)

- `sdpaSpecToPayload(spec)` (`src/adapters/sdpa/SdpaPayload.cpp:10`) emits a dict:
  `batch`, `shape{head_size, num_query_heads, num_kv_heads}`, `dtype`,
  `mask_mode`, `seqlen_q/k`, the paged/varlen lanes
  (`is_paged/block_size/is_varlen/sliding_window/use_sinks`), and the nested
  `knobs`. It **deliberately omits strides, scale_log2, softcap** — those are
  launch-time, not codegen inputs.
- Over the pybind bridge:
  `compile_service.compile("sdpa_fmha_fwd_unified", payload, arch)` →
  `_compile_sdpa_fwd_unified` (`compile_service.py:1088`) →
  **`_tiled_2d_impl(arch)`** selects the arch variant (gfx942 → the narrow-atom
  variant) → `_unified_tiled_spec_from_problem` → `UnifiedAttention2DTiledSpec`
  → **`build_unified_attention_2d_tiled(spec, arch)`** (the kernel IR; narrow
  16x16x16 atoms + strided-V on gfx942) → **`compile_kernel`**
  (`ck_dsl/helpers/compile.py:79`) lowers IR → LLVM → **HSACO** via comgr.
- Returns a wire dict: `hsaco` bytes, `kernel_name`, **grid**
  `(num_kv_heads, total_q_blocks, 1)`, **block** `(64*num_warps, 1, 1)`,
  `lds_bytes = 0` (static LDS lives in the HSACO kernarg descriptor), and
  **`arg_schema`** — the **18-slot ABI manifest**
  (`_sdpa_fwd_unified_arg_schema()`, `compile_service.py:287-394`).
  `dictToArtifact` turns it into a C++ `KernelArtifact`
  (`src/python/CompileServiceBridge.cpp:184`).

## 8. Cache + module load

- **Cache key** = `GraphSignature::computeForSpec(opKind, spec, arch)`
  (`SdpaFwdPlanBuilder.cpp:301`; folding logic `GraphSignature.cpp:147-237`).
  Folds: version + opKind + arch + B/Hq/Hkv/Sq/Skv/D + dtype/mask + paged/varlen
  lanes + **all 9 knobs**. Intentionally **NOT** strides or scale_log2.
- `JitCache::getOrLoad(key, loader)` (`runtime/JitCache.cpp:14`) — **in-memory**
  `unordered_map<hash, shared_future<HipModule>>`; the compile loader runs only
  on a miss. A hit returns the already-loaded module — no recompile.
- On a miss, the `HipModule` ctor does `hipModuleLoadData` +
  `hipModuleGetFunction` once and **captures grid/block/lds/arg_schema** off the
  artifact — `runtime/HipModule.cpp:34-69`.
- `SdpaFwdPlan` is constructed with the module + tensor UIDs + scalars; its ctor
  **validates the schema is exactly the 18-slot ABI** (slots 0-9 pointers,
  10-14 f32, 15-17 i32) — `SdpaFwdPlan.cpp:108-146`.

## 9. Execute → launch (the arg-marshalling ABI)

`SdpaFwdPlan::execute(variantPack, workspace, stream)` — `SdpaFwdPlan.cpp:187`:

1. resolve device pointers by UID (O, Q, K, V, + optional sink).
2. varlen: D2H copy of seqlens; marshal host i32 arrays (`block_tables`,
   `cu_seqlens_q`, `seqused_k`, `block_table_stride`) into the caller's
   **workspace** (three 256B-aligned regions); H2D upload.
3. recover raw scale: `rawScale = scale_log2 * ln2`.
4. **build the 18 `ArgValue`s in schema order** (`:361-380`):
   - pointers (0-9): `O, Q, K, V, sink, block_tables, seqused_k, alibi=null,
     qq_bias=null, cu_seqlens_q`
   - f32 (10-14): `rawScale, k_scale=1, v_scale=1, out_scale=1, softcap=0`
   - i32 (15-17): `num_seqs, block_table_stride, qq_bias_stride_0=0`
5. `LaunchAbi::pack(schema, values)` (`LaunchAbi.cpp:102`) lays them out at
   natural alignment, validating count/tag/align per slot.
6. `HipModule::launch(packed, grid, block, ldsBytes=0, stream)` →
   `hipModuleLaunchKernel` with the `HIP_LAUNCH_PARAM_*` extras —
   `HipModule.cpp:122-141`.

### Keeping the kernel-arg order in sync

There is **no single shared artifact file**; consistency is enforced by **three
hand-mirrored lists kept structurally identical**, plus runtime validation:

- **Ground truth** — the DSL kernel's 18 `b.param(...)` declarations, in order:
  `instances/gfx942/attention_tiled_2d.py:945-984`.
- **Python manifest** — `_sdpa_fwd_unified_arg_schema()`
  (`compile_service.py:287-394`); ships in the artifact → becomes
  `KernelArtifact.argSchema` / `HipModule.argSchema()`.
- **C++ packer order** — the `ArgValue` list in `execute()`
  (`SdpaFwdPlan.cpp:361-380`).
- **Drift guards** — the plan ctor's per-slot kind check
  (`SdpaFwdPlan.cpp:120-146`) and `LaunchAbi::pack`'s size/tag/align validation,
  which name the offending slot. A mismatch fails loudly at pack time rather than
  corrupting the launch.

---

## Two things worth internalizing

- **Compile-time vs launch-time split.** Shape + dtype + paged/varlen + the
  dispatcher's **knobs** are baked into the HSACO and folded into the cache key.
  **Scale, strides, pointers, num_seqs** are launch-time args (the 18-slot ABI)
  and are *not* in the cache key — so one compiled kernel is reused across
  different scales / strides / batch pointers.
- **Where each decision lives.** Causal / paged / scale / dtype / GQA →
  `SdpaAdapter::buildSpec` (C++, from the graph). Which kernel *variant* →
  `_tiled_2d_impl(arch)` (Python). Which *config* (knobs) → `selectPerfKnobs`
  dispatcher (C++). The actual *codegen* → `build_unified_attention_2d_tiled`
  (CK DSL). Everything reconverges on the 18-slot ABI at launch.

## Caveats / not fully traced

- The backend heuristic that ranks `ck_dsl_sdpa_engine` against *other*
  providers for the op lives in the backend engine-selection / heuristics layer
  and is not traced here.
- On the current path several launch values are fixed constants, not
  graph-driven: `k_scale/v_scale/out_scale = 1.0`, `softcap = 0.0`,
  `qq_bias_stride_0 = 0`, alibi/qq_bias = null. fp8-dequant / softcap / bias
  lanes are structural-only today. Causal is unconditional (non-causal is
  declined in `SdpaAdapter::buildSpec`).
