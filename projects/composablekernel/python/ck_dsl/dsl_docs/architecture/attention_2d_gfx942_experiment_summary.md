# CK DSL Unified Attention 2D — gfx942 (MI300X) Experiment Summary

Date: 2026-06-03 (living document — append per experiment)
Scope: fp16/bf16 prefill-2D unified attention SDPA-fwd on **gfx942 (CDNA3 / MI300X)**, the
narrow `16x16x16`-atom variant (`instances/gfx942/attention_tiled_2d.py`). Companion to the
gfx950 study `attention_2d_experiment_summary.md`; the gfx950 perf playbook largely does **not**
transfer (the wide-K `16x16x32` / `32x32x16` atoms and the transposed/combo ladder are
gfx942-illegal, rejected in `__post_init__`).

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
