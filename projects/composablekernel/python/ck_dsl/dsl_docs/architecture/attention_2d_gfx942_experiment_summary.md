# CK DSL Unified Attention 2D — gfx942 (MI300X) Experiment Summary

Date: 2026-06-03 (living document — append per experiment)
Scope: fp16/bf16 prefill-2D unified attention SDPA-fwd on **gfx942 (CDNA3 / MI300X)**, the
narrow `16x16x16`-atom variant (`instances/gfx942/attention_tiled_2d.py`). Companion to the
gfx950 study `attention_2d_experiment_summary.md`; the gfx950 perf playbook largely does **not**
transfer (the wide-K `16x16x32` / `32x32x16` atoms and the transposed/combo ladder are
gfx942-illegal, rejected in `__post_init__`).

The distilled, arch-specific lessons from this log (ISA constraints,
no-transpose-read operand feed, occupancy + compile-time budgets) are promoted
to `optimization/gfx942_playbook.md`; the kept/reverted-lever ledger is in
`optimization/runbook_compliance.md`. This file is the full per-experiment log.

**Status legend:** `KEPT` (merged, net-positive) · `REVERTED` (measured, net-negative) ·
`IN PROGRESS` (under measurement) · `DEFERRED` (out of scope / blocked on a precondition).

---

## Baseline

The gfx942 path was implemented + correctness-validated separately (see
`project-gfx942-tiled-attention`). This study covers the **performance** push on top of that:
correctness must stay **6 PASS / 2 SKIP (D256) / 0 FAIL** vs `CpuFpReferenceSdpa`, and gfx950
stays byte-identical (all kernel edits live in `instances/gfx942/`).

### The binding constraint (Phase 0 diagnosis)

gfx942 = 64 KB LDS / CU. `probe_occupancy --arch gfx942` confirms the entire admitted config
family is **LDS-bound at exactly 1 CTA/CU**; spill=0 everywhere, VGPR/AGPR/SGPR never bind.

| config | VGPR/lane | LDS B | CTAs/CU | limited_by |
|---|---:|---:|---:|---|
| D128 nw2/mw16/t64 (D128 ship cfg) | 231 | 61,952 | 1 | LDS |
| D64 nw4/mw32/t64 (D64 ship cfg) | 134 | 51,200 | 1 | LDS |
| D64 nw2/mw32/t32 (B32 ship cfg) | ~ | ~30,720 | 2 | LDS |

- **2-CTA/CU threshold on gfx942 = ≤ 32,768 B/CTA** (64 KB / 2).
- LDS blocks (bpe=2): `K_lds = 2·T·HD·2` (double-buffered, dominant on D128 ~32 KB),
  `V_lds = T·HD·2`, `P_lds = BLOCK_M·(T+8)·2`, `Acc_lds = BLOCK_M·OUT_STRIPE·2`.
- `register_pv` removes `P_lds`: at D64 nw4/mw32/t64 that is 18,432 B → 51,200 − 18,432 =
  **32,768 = exactly the 2-CTA line.** This is the cleanest single path to 2 CTAs/CU.
- D128 maxes at nw2/t64 (nw4/nw8 D128 are LDS-rejected by the dispatcher gate); reaching 2
  CTAs/CU on D128 would require K single-buffering *and* `register_pv` — the hard case.

### Scoreboard at baseline (pre-fix → after the kept selector fixes)

TFLOPS, causal, `4·B·Hq·S²·D`, median/50. `analytic` is what ships on gfx942 (the ML model is
gfx950-trained → `ml=UNAVAILABLE`; `selectAnalyticFallback` is the gfx942 path). `oracle` = the
best enumerated config per shape; `PyTorch` = aotriton flash (TheRock gfx94X nightly).

| Shape | dtype | analytic (pre) | analytic (now) | oracle | PyTorch | now %PT |
|---|---|---:|---:|---:|---:|---:|
| GQA S2048 D64 | f16 | 83.8 | **142.6** | 142.6 | 260 | 55% |
| InFam GQA8 D64 S2048 | bf16 | 84.2 | **165.3** | 165.3 | 303 | 55% |
| InFam GQA8 D64 S2016 B32 | bf16 | 124.3 | **165.1** | 165.4 | 256 | 64% |
| GQA S2048 D128 | f16 | 111.6 | **115.7** | 115.8 | 293 | 40% |
| GQA S4096 D128 | f16 | 124.8 | **127.4** | 127.6 | 280 | 46% |
| MHA S2048 D128 | f16 | 108.5 | 108.3 | 108.2 | 292 | 37% |
| GQA B4 S2048 D128 | f16 | 123.6 | **127.6** | 127.6 | 422 | 30% |
| Bf16 GQA S2048 D128 | bf16 | 107.9 | **111.3** | 111.4 | 289 | 39% |
| GQA B8 S2048 D128 | f16 | 125.4 | **128.5** | 128.5 | 421 | 31% |
| GQA B4 S4096 D128 | f16 | 127.3 | 127.3 | 127.5 | 301 | 42% |
| GQA S8192 D128 | f16 | 128.2 | 128.5 | 128.8 | 532 | 24% |

Geomean analytic ≈ **35% → 41% of PyTorch** after the two selector fixes; analytic now tracks
the oracle ceiling cohort-wide. **The selection gap is closed; all remaining headroom is the
kernel ceiling**, and the ceiling gap is widest on the large-S D128 shapes (S8192 = 24% of PT).

---

## Methodology — the optimization loop

Two nested loops. The **inner loop** runs one experiment; the **outer loop** generates the next set.

**Inner loop (one lever):**
`hypothesize → verify correctness (6/2/0) → measure (probe_occupancy + oracle vs PyTorch) →
keep or revert → explain via probe/ISA diff → record here.`
Non-negotiable: correctness gates before perf; no verdict is recorded without a mechanistic
explanation (the *why* is what feeds the outer loop). One lever per change.

**Outer loop (what to try next) — the bound is the generator:**
1. **Re-diagnose the bound after every kept change** (`probe_occupancy`, then rocprof). The bound
   names the eligible levers: while `limited_by == LDS`, only LDS-cut / occupancy levers move the
   ceiling — scheduling hints are a priori dead.
2. **Mine the last experiment's evidence — especially negatives.** Each measurement exposes a new
   axis (e.g. swizzle exposed that VGPR is tight on nw4/mw32; that now informs every nw4/mw32 lever).
3. **Consult the gap map** (oracle vs PyTorch) to bias toward where headroom is concentrated.
4. **Completeness critic:** what shape / knob / modality is untested? (deferred levers re-enter when
   their precondition is met).
5. **Rank survivors by `impact × independence ÷ risk`; batch to the 3-worktree budget.**

Measurement mechanics (gfx942): static probes (`probe_occupancy`, `probe_isa_inspect`) run
per-worktree with no build (they import `ck_dsl` from cwd). The correctness/oracle gtest binary
bakes the **feature** worktree's `ck_dsl` path (`CkDslProviderPaths.cmake`; `PYTHONPATH` ignored),
so authoritative on-device measurement of a WIP kernel edit is done by copying the edited file into
the feature tree, running, then `git checkout --` to restore — inherently serial on the one GPU.

---

## Lever ledger

| Lever | Kind | Status | Headline result | Bound at time | Next hypothesis spawned |
|---|---|---|---|---|---|
| S1 D64 mw=32 + 1× tile | selector | **KEPT** | 3 D64 shapes → oracle (1.33–1.96×) | LDS / selection | block_size keys nw (4 if bs≥64 else 2) |
| S2 D128 early-V | selector | **KEPT** | +2–4% on 6 GQA D128 | LDS / selection | early-V inverts at S8192; gate seqlen≤4096 |
| X P/Acc XOR swizzle | kernel | **REVERTED** | D128 noise; bf16 D64 −14.7% | LDS @1 CTA/CU | VGPR tight on nw4/mw32 → watch reg pressure |
| R register-PV (bf16 D64) | kernel | **MARGINAL** | 2 CTA/CU reached, but only **+2.6%** (1 shape) | LDS @1 CTA/CU | occupancy thesis spent → pivot to intra-CTA compute |
| A async LDS-DMA 1→2 dword | kernel | **REVERTED (dead)** | b64 IR-illegal; b96/b128 abort comgr | LDS @1 CTA/CU | no wider async-LDS-DMA exists on CDNA3 |
| G agpr-alloc decouple | kernel | **DEFERRED (likely dead)** | not VGPR-bound | LDS @1 CTA/CU | bound never shifted to VGPR → no trigger |
| K single-buffer (D128) | kernel | **DEFERRED (likely dead)** | frees ~16 KB; HIGH risk | LDS @1 CTA/CU | R showed 2 CTA/CU buys only ~2.6% → not worth the risk |

---

## Experiment results

### S1 — D64 plain-atom mw=32 + 1× tile selection [KEPT]

Goal: the analytic selector hardcoded `block_m_per_warp=16` + 2×-tile, so the oracle-best D64
`nw4/mw32/t64` plain-16x16 candidate (enumerated) never won.

Result: KEPT. All three D64 shapes move to oracle. Commit `660d6e2c0ff`.

Findings:
- On gfx942 D64, oracle is **always mw=32** with a **1× tile** (`tile_size == block_size`). The
  mw=32 + 2×-tile combo (nw4/mw32/t128 ≈ 70 KB) is LDS-rejected; the 1× combo (≈ 51 KB) fits.
- `num_warps` keys on **block_size, not batch**: bs≥64 → nw4 (BLOCK_M=128 amortizes T=64 KV/iter),
  bs=32 → nw2. (The oracle case name `…S2016_B32` is mislabeled — its `batch` field is 1.)
- The gfx950 comment that plain-atom mw=32 is a "latent trap" is a gfx950 fact (4→2 CTA/CU loss);
  on gfx942 we are already at 1 CTA/CU, so that reasoning does not apply.

Evidence: oracle `analytic=` jumped 84.3→142.6 / 84.2→165.3 / 124.3→165.1, each landing on the
shape's `best=`. Arch+head gated; gfx950 / D128 untouched.

Action: shipped in `SdpaCandidateSelector.cpp analyticTarget` + mirror in DSL
`_select_2d_block_m_per_warp`.

### S2 — D128 GQA early-V schedule selection [KEPT]

Goal: `early_v` (overlaps V load with QK+softmax, no LDS cost) is the oracle-best flag on the GQA
D128 shapes, but `analyticCloseness` penalized it, so the analytic picked the plain variant.

Result: KEPT. +2–4% on 6 GQA D128 shapes. Commit `5e1ecbe45cb`.

Findings:
- Two carve-outs from oracle data: **MHA** D128 oracle-best keeps the plain schedule (gate to
  GQA, `num_queries_per_kv>1`); and the benefit **inverts at S8192** (V load already well-hidden) —
  the first cut regressed S8192 ~1%, fixed by gating `seqlen_q ≤ 4096`.

Evidence (analytic, post-S1 → post-S2): S2048 D128 111.6→115.7, S4096 124.8→127.4, B4_S2048
123.6→127.6, bf16 S2048 107.9→111.3, B8_S2048 125.4→128.5. MHA and S8192 unchanged.

Action: `prefer_early_v` target field set for gfx942 D128 GQA seqlen≤4096; closeness rewards
early-V there, penalizes it elsewhere.

### X — P_lds / Acc_lds XOR swizzle [REVERTED]

Goal: remove LDS bank conflicts (and the `+8` P_lds pad) via XOR swizzle on the 64 KB part.

Result: REVERTED — net-negative. WIP-X not merged.

Findings:
- The epilogue **global store was already wide** (`global_store_dwordx4` / fp16x8) — the Phase-0a
  "narrow vmem_store=4" was the count, not a width problem. Epilogue vectorization is a no-op here.
- The swizzle's per-row address recompute pushed **VGPR 143 → 228** on the bf16 D64 nw4/mw32 path
  and tanked it; the freed ~2 KB LDS does not cross the 2-CTA threshold, so no occupancy offset.

| Shape | swizzle off → on | verdict |
|---|---|---|
| Bf16 InFam GQA8 D64 S2048 | 165.3 → 141.0 | **−14.7%** |
| D128 cohort | within ±0.8% | noise |

Action: reverted. Carries forward the insight that **VGPR is tight on nw4/mw32** — a constraint
for every nw4/mw32 lever (and a reason `agpr` (G) is likely worthless while LDS-bound).

### R — register-PV on bf16 D64 [MARGINAL — measured, not shipped]

Goal: drop `P_lds` to reach **2 CTAs/CU** on bf16 D64 nw4/mw32/t64 (51,200 → 32,768 B) and test
whether 1→2 CTA/CU converts to TFLOPS or the `_permute_p_c_to_a16` cross-lane reshape eats it.

Result: **2 CTAs/CU confirmed, but the win is only +2.6% on one shape.** Measured by adding a
bf16 `register_pv` candidate to the C++ enumerator and running the oracle (change reverted after
measurement — not shipped).

| bf16 D64 shape | regpv=0 best | regpv=1 best | verdict |
|---|---:|---:|---|
| InFam GQA8 D64 S2048 (nw4/mw32/t64, was 1 CTA/CU) | 165.0 | **169.3** | regpv wins **+2.6%** |
| InFam GQA8 D64 S2016 B32 (nw2/mw32/t32, already 3 CTA/CU) | 166.6 | not picked | regpv loses |

Findings:
- Occupancy probe confirmed the LDS math exactly: nw4/mw32/t64 51,200 → 32,768 B, **1 → 2 CTA/CU**,
  VGPR unchanged at 143 (no blowup, unlike swizzle).
- But the reshape is heavy: register_pv adds **+129 `ds.bpermute` + ~384 `ds.swizzle`** (IR doubles,
  3,241 → 6,475 lines). The shuffle-ALU consumes most of the 2× occupancy → net +2.6%.
- register_pv only helps configs actually stuck at 1 CTA/CU (nw4/mw32/t64). For nw2/mw32/t32 (already
  3 CTA/CU) the occupancy gain is moot and the reshape is pure cost → it loses.

**Decisive diagnostic:** going from 1→2 CTAs/CU buys only ~2.6%, so the kernel is **compute/reshape-
bound within the CTA, not occupancy/latency-bound**. This retires the "raise occupancy" thesis as
the dominant lever and keeps D128 K-single-buffering dead (it only mattered if 2-CTA paid big).

Action: not shipped (the +2.6% on one shape requires selector plumbing + `_permute_p_c_to_a16`
correctness verification on the gfx942 narrow lane map + unit-test churn — poor effort/reward).
Reproducible via the enumerator one-liner if revisited.

### A — async LDS-DMA width 1 → 2 dword [REVERTED — architectural dead-end]

Goal: if gfx942 legalizes **b64** (2-dword) load-to-LDS, halve the async-DMA call count
(≈4× heavier on D128).

Result: **DEAD.** The IR builder rejects 2-dword outright:
`async_buffer_load_lds_addr dwords must be 1, 3, or 4 (got 2)`. b64 is not expressible; the 3/4-dword
(b96/b128) widths are exactly the documented CDNA3 comgr-abort cases. There is no wider legal
async-LDS-DMA on gfx942 — width=1 is the only viable setting. No code changed.

---

## Current best policy (gfx942 analytic selector)

- **D64** (head_size=64): `block_m_per_warp=32`, `tile_size=block_size` (1×), `num_warps = bs≥64?4:2`
  then DMA-floor clamp; plain async pipeline.
- **D128** (head_size=128): `block_m_per_warp=16`, `tile_size=2×block_size` → nw2/t64; GQA with
  `seqlen_q≤4096` adds `use_early_v_schedule`; MHA and S8192 keep the plain schedule.
- All schedule/atom variants otherwise off (gfx950-only atoms are illegal). D256 unsupported (LDS).

## Current gaps to PyTorch

- D64: analytic ≈ oracle ≈ **55–64% of PyTorch** (selection gap closed; ceiling is the limit).
- D128: **24–46% of PyTorch**, gap widest at large S (S8192 = 24%) — our kernel plateaus flat
  (~108–128 TF) while PyTorch flash scales with size. This is the LDS-occupancy ceiling, not a
  selection problem.

## Recommendations / next levers (bound-driven)

**Batch 1 outcome (the pivot):** the cheap **selector** levers delivered the gains (S1+S2: analytic
35% → 41% of PyTorch, now tracking the oracle ceiling cohort-wide). The **kernel** levers did not
move the ceiling: X (swizzle) net-negative, A (async-DMA) architecturally dead, R (register-PV)
only +2.6% on one shape. R's verdict is the important one — **1→2 CTA/CU buys ~2.6%, so the kernel
is compute-bound within the CTA, not occupancy-bound.** The occupancy-raising branch is therefore
largely spent on this narrow-16x16 path, and the remaining D128 gap (24–46% of PyTorch) is a
genuine architectural limit (gfx942 cannot use the wide-K / 32x32 atoms that carry the gfx950 wins).

Next, in order:
1. **rocprof the compute bound.** Since occupancy is not the limiter, profile the hot loop
   (`MemUnitStalled`, `VALUBusy`, MFMA issue stalls, LDS bank conflicts via `SQ_LDS_BANK_CONFLICT`)
   on the D128 ship config to find what actually caps the per-CTA throughput. This re-seeds the next
   experiment set with evidence rather than hypothesis.
2. **Intra-CTA pipelining** (the live branch now that occupancy is spent): early-V *depth*, K/V
   prefetch distance, QK↔softmax↔PV overlap scheduling — measured against the rocprof bound.
3. **Ship decision on R (+2.6%, bf16 D64 nw4/mw32/t64):** only if the marginal win justifies the
   selector plumbing + `_permute_p_c_to_a16` correctness verification + unit-test churn. Currently
   parked as not-worth-it.
4. **G (agpr) / D128 K-single-buffer: closed** — both were occupancy plays; R showed occupancy
   barely pays. Do not invest without a new bound that re-opens them.

---

## Batch 2.1 — rocprof the bound [done; supersedes the "compute-bound" framing above]

rocprofv3 on the ship configs (logs `WIP/.../rocprof/`). The bound is **LDS bank conflicts**, not
compute — correcting the post-R wording: the MFMA pipe is ~95% idle because warps stall on LDS.

| metric | D128 (40% PT) | D64 (55% PT) |
|---|---:|---:|
| MFMA pipe busy | **4.63%** | 6.64% |
| busy cycles waiting on LDS (`SQ_WAIT_INST_LDS/BUSY`) | **0.574** | 1.580 |
| bank conflicts / LDS instr | **11.34** | 6.61 |
| LDSBankConflict (derived) | 40.3% | 30.0% |
| MemUnitStalled | ~0% | ~0% |
| inst mix | VALU 54% / LDS 20% / MFMA 7% | VALU 70% / LDS 14% / MFMA 5% |

Diagnosis: **LDS-bank-conflict bound** (D128: 11.3 conflicts/LDS inst, ~57% of busy spent waiting on
LDS), MFMA-starved, softmax-VALU-heavy, NOT memory- or occupancy-bound. This also explains R
(2 CTA/CU doesn't relieve per-access conflict serialization) and X (it swizzled `P_lds`/`Acc_lds`,
but the hot conflict is the **strided-V B-operand LDS read** — the gfx942 narrow path emulates the
`ds_read_tr16` lane map with strided loads; X targeted the wrong tile, hence D128 unmoved).

**Next experiment (Batch 2.2):** targeted de-conflict of the strided-V (and K_lds) read — a swizzle
or layout change on that specific access, with cheap addressing to avoid the VGPR blowup that sank
the blanket swizzle (X). Validate with the same rocprof counters: `SQ_LDS_BANK_CONFLICT` per LDS
inst and MFMA-busy should rise on D128.

---

## Batch 3 — the flash target, the first kernel win, and the LDS dead-end

**Flash is the proof there is huge room — and the gap is one design decision.** rocprofv3 on PyTorch
aotriton `attn_fwd`, D128, same counters as ours:

| D128 | flash `attn_fwd` | ours | 
|---|---:|---:|
| bank conflicts / LDS-inst | **0.58** | 11.34 |
| LDSBankConflict (derived) | **1.0%** | 40.3% |
| MFMA-util | 14.3% | 4.6% |
| LDS-wait / busy | 0.12 | 0.57 |
| MemUnitStalled | 20.9% (HBM-bound — the *right* bound) | ~0% |

Flash is essentially conflict-free and near the HBM roofline; we stall ~57% on self-inflicted V-read
conflicts. The entire gap is the strided-V emulation. (Attention is softmax-VALU-heavy, so even flash's
MFMA-util is only 14% — MFMA was never the prize; *not stalling on LDS* is.)

**V_lds padding [KEPT, commit `a7dcd8af487`] — first kernel win.** A read-side XOR is infeasible (async
DMA writes lane-contiguous; no per-lane dest). Padding the V_lds row stride (+8 halves) — the only
DMA-compatible lever — drops the D128 read 4-way → 2-way. **+2–5% on every D128 shape** (GQA S2048
+4.6%), VGPR-neutral (226→230), correct 16/2/0 on the expanded net. D64 auto-disabled (can't help:
8 rows/DMA-call). Env-gated `HIPDNN_GFX942_SWIZZLE_VLDS` (default on); gfx950 byte-identical.

**L1 conflict-free V LDS layout [REVERTED — proven net regression, not shipped].** Goal was a wide
bank-spread `ds_read_b64` from a re-laid-out V_lds. Conflict-cycle accounting (store+read/kv-iter):
baseline 256 → committed pad **128** → best wide-layout 192 → reshape 704. **No full design beats the
committed pad**, because (1) the async DMA can't produce a col-major layout (only the wave-uniform base
is free — which the pad already uses), and (2) the transpose is symmetric and **gfx942 has no
`ds_read_tr_b16`** (the fp8 stripe precedent works only because `ds_read_tr_b8` exists). Element-degree-1
is also mathematically impossible (64 words / 32 banks = 2/bank floor; CK's `MakeVLdsBlockDescriptor`
pays it too). **The committed padding is the Pareto-optimal LDS-only lever on this ISA.** Analysis:
`WIP/.../WIP-L1/HANDOFF_FINDINGS.txt`.

**Corrected understanding:** flash's advantage is NOT a clever V *LDS* layout — it's **not round-tripping
V through LDS at all.** Closing the gap requires **L2: register-resident V for PV** (CK `vr` pipeline) —
removes the V LDS read entirely (no transpose, no store conflict). Favorable: we're LDS-bound at 1
CTA/CU with VGPR headroom (~176/512), and dropping V_lds also frees LDS. Risk: VGPR pressure could wall
it. **Next: feasibility-check L2's VGPR budget before committing to the rewrite** (same discipline that
caught L1). L3 (32x32x8 atom) and L4 (prefetch) remain second-order until V is off the LDS-stall path.

**L2 register-V + in-register shuffle transpose [REVERTED — flat, lgkmcnt serialization].** Feasibility
passed (VGPR *lower* 240→176, V is the whole bound, partial-rewrite scope). Prototype (D128, flag
`HIPDNN_GFX942_REGISTER_V`) made the V read conflict-free (1 `ds_read_b64` at the floor, `ds_read` −73%)
and is correct (16/2/0 on the expanded net). BUT the in-register transpose's **+256 `ds_swizzle` are
LDS-port ops that serialize on `lgkmcnt` (18→234)** — on-device D128 = **flat to −1.4% vs the committed
padding.** The static red flag was correct: the swizzle serialization costs what the conflict-removal
saves. (Strategy-2 — half the swizzle moved to VALU `v_perm` — is the only untried refinement; bounded
upside.)

**Conclusion: the V-read bound is extracted to its gfx942 ceiling.** With no `ds_read_tr_b16`, every
transpose approach hits *either* bank conflict (strided) *or* `lgkmcnt` serialization (swizzle); the
committed V_lds padding (+2–5%) is the Pareto frontier. The remaining gap to flash (2.5×) is **structural
pipelining** (deep K/V prefetch + full QK·softmax·PV overlap, flash's regime), not a V-read lever — a
larger rewrite. **Ledger: 3 wins shipped (D64 sel, D128 early-V, V padding → analytic 35%→41% of PT);
6 levers killed with proof (X, A, R, L1, L2, occupancy branch).**

---

## Batch 4 — the flash-pipeline rewrite (v2)

Batch 3 concluded the gap to flash was "structural pipelining, a larger rewrite." Batch 4 **is**
that rewrite: a fresh 32x32x8 wide-atom flash pipeline on the gfx942 path, built correctness-first
(Stream A foundation) then layered with the levers below. All edits are env-gated
(`HIPDNN_GFX942_FLASH_PIPELINE` level 0–5), default OFF; gfx950 byte-identical by construction.
Merged at commit `9cff43bdd2d`.

### Phase-A correction — what flash actually does (re-profiled)

**The v1 ledger's claim that "flash never round-trips V through LDS at all" is WRONG — corrected
here.** Fresh kernel-trace + counters of PyTorch aotriton `attn_fwd` on gfx942 (MI300X), D128 fp16:

| flash `attn_fwd` (D128 fp16) | value |
|---|---:|
| LDS (dynamic) | **32,768 B** (V *is* staged in LDS) |
| bank conflicts / LDS-inst | **0.58** (conflict-free) |
| LDSBankConflict (derived) | 1.0% |
| archVGPR / accumVGPR | **128 / 384** (acc lives in AGPR) |
| SGPR / scratch / WG | 112 / 64 / 256 (num_warps=4) |
| num_stages | 1 |
| occupancy | ~1 wg/CU (32 KB LDS, high AGPR) |
| MFMA util | 14.1% |
| MemUnitStalled | 21.8% (HBM-bound — the *right* bound) |

(D64 variant: 16,384 B LDS, archVGPR 128 / accumVGPR 128 — same num_warps=4, smaller-tile/low-AGPR.)

So flash **does** put V in LDS. Its edge is not deep prefetch (num_stages=1) and not high occupancy
(~1 wg/CU, same as our v1): it is **(a) a conflict-free V LDS read (0.58 vs our 7–11 conflicts/inst)**
and **(b) operand residency — P and acc are kept off LDS** (acc in AGPR, no P_lds bridge). The kernel
is HBM-bound, the correct roofline. This re-pointed the whole v2 plan at *operand residency +
conflict-free V*, not occupancy or prefetch depth.

### Lever ledger (Batch 4)

| Lever | Kind | Status | Headline result | Mechanism / why |
|---|---|---|---|---|
| 32x32x8 base (Stream A) | kernel | **KEPT (foundation)** | correctness-first wide-atom; ~2× *slower* than narrow alone (naive V) | wider K=8 atom alone cut conflicts 11.34→7.25/LDS-inst, but the naive strided V keeps it LDS-bound (MFMA 7%) |
| transposed-x8 / register-P^T (Stream C) | kernel | **KEPT** | **+1.57×** over plain-x8 (up to 1.82× @ S8192); −62% LDS insts | S^T = K·Q^T so P^T is the PV B-operand *direct from registers* → P_lds round-trip eliminated (SQ_INSTS_LDS 767K→290K) |
| FA-4 rescale-skip (Stream D) | kernel | **REVERTED (flat)** | within ±1% on 5/7, slightly neg on 2 | DSL has no warp-ballot / predicated `scf.if`-with-results → the `o_acc *= alpha` rescale still issues (runtime `alpha`, backend can't fold `x*1.0`); only P's frame changes |
| conflict-free-V (Stream prong3) | kernel | **BLOCKED (incorrect)** | cfv as MFMA-**A** on transposed path → 12/9 FAIL (S512 finite 0.165 err) | operand-role asymmetry: [HD,T+pad] is the natural MFMA-**B** layout; cfv-as-B is proven correct (recovery L1), cfv-as-A needs an in-register MFMA-A lane reshape this kernel doesn't do |
| acc→AGPR / drop Acc_lds | kernel | **REJECTED (evidence gate)** | not implemented | STEP-0 proved the 48 KB peak = K_lds+V_lds only; the backend LDS-coalescing pass already aliases Acc_lds into the loop-dead K/V region → dropping it is moot for occupancy |
| K single-buffer (prong2) | kernel | **KEPT + SHIPPED** | **L4: 156–178 TF, +18–33% over baseline, 2 wg/CU** | K double→single buffer drops LDS 48→32 KB, crossing the 32 KB threshold → 1→2 wg/CU |

### The valid perf ladder (fp16 D128)

Levels are cumulative env settings. base = the narrow 16x16x16 ship path (L0). Geomean over the
7 flagged fp16-D128 shapes (Stream C job 354226 / prong2 job 354431); flash ≈ 290 TF.

| level | path | geomean / representative TFLOPS | vs base | vs flash |
|---|---|---:|---:|---:|
| L0 | narrow 16x16x16 baseline (ship) | **126.6** | 1.00× | 0.44× |
| L1 | plain-x8 (32x32x8, P_lds bridge, naive V) | 59.5 | 0.47× | 0.21× |
| L2 | transposed-x8 (register-P^T, naive V) | 93.3 | 0.74× | 0.32× |
| **L4** | **transposed-x8 + K single-buffer** | **156–178** | **+18–33%** | **~0.62×** |

Plain-x8 (L1) is the correctness-first foundation and is *below* the narrow baseline — the naive
strided V plus the P_lds bridge dominate. The transposed register-P^T handoff (L2) recovers most of
that deficit (+57% over L1), and K single-buffering (L4) crosses the occupancy threshold to land
**above** the narrow baseline. Per-shape L4 (prong2/prong3 medians):

| shape (D128 fp16) | L0 base | L2 (x8) | **L4 (x8+k1buf)** | L4 vs base |
|---|---:|---:|---:|---:|
| GQA S2048 | 120.9 | 81.8 | **147.7** | +22% |
| GQA S4096 | 134.4 | 91.2 | **165.8** | +24% |
| MHA S2048 | 113.3 | 91.3 | **132.9** | +17% |
| GQA B4 S2048 | 133.7 | 90.2 | **171.5** | +28% |
| GQA S8192 | 135.0 | 113.7 | **178.9** | +33% |
| GQA B8 S2048 | 134.9 | 91.9 | **174.6** | +29% |
| GQA B4 S4096 | 137.3 | 95.6 | **173.0** | +26% |

Best observed: **178.9 TF (GQA S8192) ≈ 62% of flash.** D64 fp16 and all bf16 are controls (flag
inert) and stay flat across levels — measurement validated. **L4 ships: the analytic selector
auto-selects it for gfx942 D128 fp16 (commit `9cff43bdd2d`).**

### Mechanism for L4 (K single-buffer → 2 wg/CU)

STEP-0 (static gate) decomposed the 48 KB peak exactly: `K_lds = 2·T·HD·2 = 32 KB` (double-buffered),
`V_lds = T·HD·2 = 16 KB` (already single-buffered), `P_lds = 0` (register-P^T), `Acc_lds` aliased
(see rejected lever above). The only lever that crosses the 32 KB / 2-wg threshold is cutting loop LDS,
and V is already single-buffered — so **K single-buffer** (32→16 KB) → loop LDS = 16+16 = **32 KB →
2 wg/CU** (rocprof-confirmed: `..._earlyv` group_segment 49,152 B / 1 wg/CU → `..._earlyv_k1buf`
32,768 B / 2 wg/CU). The single-slot K introduces a read-race (next-tile async DMA write `K[i+1]` vs
the tail of `QK[i]`'s LDS reads); **closed by a post-QK `s_waitcnt(lgkmcnt=0)` + barrier** before the
next-K DMA, guaranteeing all QK K-reads retire before the shared-slot overwrite. Numerics unchanged
(same online-softmax math); the fitter enforces `BLOCK_M ≤ tile_size` for L4 (fixes the S528 multi-tile
edge case). The prior R-lever floor (1→2 wg/CU = +2.6% on a conflict-bound kernel) did *not* hold here:
on these latency-bound long-prefill shapes the doubled occupancy buys +17–33% over baseline.

### Remaining path to flash (62% → 100%)

The last gap is the **conflict-free V** read (flash's 0.58 vs our ~7 conflicts/LDS-inst). The Batch-4
prong3 work narrowed the blocker conclusively: cfv on the transposed path fails because the natural
[HD,T+pad] LDS layout is the MFMA-**B** layout, but the transposed path consumes V as MFMA-**A** —
an **operand-role asymmetry** (cfv-as-B is proven correct; only cfv-as-A fails, S512 finite 0.165 err).
The fix is a NEW feature, not a read-index fix: an **in-register MFMA-A VALU lane reshape** — port CK's
`MakeShuffledVRegBlockDescriptor`. Critical constraint: do the reshape with **`v_perm` (VALU), not
`ds_swizzle`** — ds_swizzle ops are LDS-port ops that serialize on `lgkmcnt` (exactly the trap that
made v1's L2 register-V transpose flat). This is an **active follow-on**, not yet implemented.

### Current best policy (updated for v2)

- **D128 fp16**: **L4 transposed-x8 + K single-buffer ships** (analytic auto-selects on gfx942 D128
  fp16; +18–33% over the narrow baseline, 2 wg/CU, ≈62% of flash, commit `9cff43bdd2d`). Levels 0–3/5
  and the FA-4 path remain gated OFF building blocks.
- **D128 bf16**: stays on the narrow 16x16x16 path (the x8 flash atom is fp16-only); D128 bf16 selector
  unchanged from Batch 1–2 (plain nw2/t64 + GQA early-V seqlen≤4096).
- **D64** (fp16/bf16): unchanged from Batch 1 (`mw=32`, 1× tile, `nw = bs≥64?4:2`).
- D256 unsupported (LDS). gfx950 byte-identical.

### Current gaps to PyTorch (updated)

- **D128 fp16: now ~62% of flash** (was 24–46% at end of Batch 3) — L4 shipping lifts the large-S
  shapes that previously plateaued flat (S8192 24% → ~62%). The residual 38% is the conflict-free-V
  read, blocked on the MFMA-A register reshape (active follow-on above).
- **D128 bf16: unchanged (~39% of flash)** — no wide x8 atom for bf16 yet; the v2 ladder is fp16-only.
- **D64: unchanged (~55–64% of PyTorch)** — selection gap was already closed in Batch 1; the v2 rewrite
  is a D128-only effort.

---

## Batch 5 — conflict-free V: units-bug fix, the store bottleneck, the async-DMA target

**Framing correction (supersedes Batch-4 "operand-role asymmetry / MFMA-A reshape" — that was a red
herring; and there is NO gfx942/ISA ceiling).** The remaining ~38% to flash is an **implementation gap**,
proven by two working kernels at ~290 TF (D128 fp16) on this exact silicon: **aotriton `attn_fwd`** (the
PyTorch reference we profile) and **CK-Tile `block_fmha_pipeline_qr_ks_vs`** (in this monorepo). They use
the primitives we already have (32x32x8 via 2× iterateK, register-resident operands, async DMA,
`v_perm` register transpose). Flash-level is reachable — the task is to replicate the working kernels'
V feed. Any "62% is the ceiling" reading (including this doc's Batch-3 "V-read bound extracted to its
gfx942 ceiling" line) is **wrong**; these are existence proofs.

### The cfv blocker was a units bug, not cross-lane geometry [FIXED, commit `73753189ad1`]

cfv's D128 fp16 failure (4 shapes, head-dims≥8, ~45% wrong, sign-flipped finite) was traced — via the
decisive isolate `cfv-scalar-read == cfv-wide-read` (rules out read width/addressing) + code audit — to a
**byte-vs-element units bug in the cfv HBM gather**: `_issue_v_transposed` passed the paged-KV
descriptor's **byte** offset to a *typed* `b.global_load(value=f16*, voff)`, which lowers to
`getelementptr half` (an **element** index, auto-scaled ×2) → read `V[2·linear]`. Every other K/V loader
uses `async_buffer_load_lds_addr` (raw byte offset) and was unaffected. Fix (+11 lines, gfx942/cfv-only):
divide `voff` by `KV_BYTES`. **cfv is now CORRECT 16/2/0 on the full net** (independently reproduced;
L0/L4 unchanged; gfx950 byte-identical). The Batch-4 `perm_b32` op (`5f5dd0c5be3`) + read-time reshape
(`fc905546c70`) were built chasing the red-herring theory and are harmless gated-off dead code (the
read-time path never fixed it — proof the bug was the store/gather, not the read or the operand role).

### Correct cfv is perf-negative AS IMPLEMENTED — the bottleneck is the V *store*

| config | TFLOPS (D128 fp16) | wg/CU | MFMA-util | bound |
|---|---:|---:|---:|---|
| L4 (shipped, naive V) | 148–178 | 2 | 8.3% | V-conflict / LDS |
| cfv (correct, sync-gather store) | **~30** | 1 | 4.5% | **store / HBM-stall (VMEM +25%)** |
| flash (aotriton / CK-Tile) | ~290 | 1 | 14% | HBM roofline |

The conflict-free LDS **read** goal was achieved (rocprof: cfv LDS-inst halved, LDS-wait ≈0). But cfv's
**store** transposes V via a **synchronous per-element `global_load` gather** — that VMEM gather stalls
on HBM and is ~5× slower than L4's pipelined async HW DMA. So cfv-as-implemented is a net loss; **L4
remains the shipped winner.** The conflict-free read is right; only the store vehicle is wrong.

### The target = the working kernels' V feed: async-DMA + in-register transpose [ACTIVE — tackling now]

flash/CK never gather V transposed and never sync-store. They **async-DMA V in NATURAL [token,dim]
layout (pipelined HW DMA) → transpose in REGISTERS with `v_perm` (CK `shuffle_tile`, dispatched
*in-thread* — no cross-lane, by arranging the V tile distribution so each lane already holds its
head-dim's tokens) → store to LDS in the A-consumable layout → plain conflict-free `ds_read_b64`.** Port
this: replace cfv's sync gather with async-DMA(natural) → `perm_b32` register transpose → LDS store.
Budget: cfv pad → ~1 wg/CU, which is fine (flash is ~1 wg/CU on D128 — conflict-free, not occupancy, is
the win). **Reference the working source directly** — CK-Tile `block_fmha_pipeline_qr_ks_vs.hpp` +
`transpose_vectors.hpp` (`__builtin_amdgcn_perm`) + `warp_gemm_attribute_mfma_impl.hpp` (gfx942 2×-iterateK
x8) — replicate its V distribution + transpose exactly; do not re-derive from theory and do not invoke an
ISA ceiling (the working kernels disprove one). This is the active path to flash.
