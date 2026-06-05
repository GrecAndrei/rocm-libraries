# gfx942 / CDNA3 Optimization Playbook

Scope: **gfx942 / CDNA3 (MI300X) only**; for gfx950 see the optimization
runbook (`optimization/optimization_runbook.md`, especially §17.4) and the
gfx950 attention study (`architecture/attention_2d_experiment_summary.md`).
This page captures arch-specific facts that do **NOT** transfer — gfx950 *has*
`ds_read_tr_b16` and the wide-K `32x32x16` / `16x16x32` atoms; gfx942 does not.
Treat every fact here as gfx942-local unless it explicitly says otherwise.

This is the gfx942 home for lessons promoted out of the
`UnifiedAttention2DTiledSpec` CDNA3 perf push. The general (DSL-wide) versions
of the loop-roll and `ds_swizzle` lessons live in the shared docs (cross-refs
below); the numbers and the no-transpose-read pattern live here.

---

## ISA constraints (gfx942)

These are the CDNA3 facts that reject the gfx950 perf ladder. They are checked
in the spec validator (`__post_init__`) where applicable.

- **No `ds_read_tr_b16` (transpose-read).** `ds_read_tr16_b64` and the i8/fp8
  `ds_read_tr_b8` are **gfx950-only**. On gfx942 there is no transpose-read
  instruction at all, so a matmul that needs a transposed LDS operand (e.g. the
  V operand of the PV matmul in attention) cannot get it on the read side — it
  must be transposed on the **store** side, in registers (see "Conflict-free
  operand feed" below).
- **b96 / b128 async-LDS-DMA abort comgr; b64 is IR-illegal.** The async
  DRAM→LDS path (`async_buffer_load_lds_addr`) is stuck at **1 dword** on CDNA3:
  b64 raises `dwords must be 1, 3, or 4` in the IR, and b96 / b128 abort comgr.
  There is no async-DMA widening lever on gfx942.
- **`32x32x8` is THE gfx942 wide f16 MFMA atom.** The wide-K `32x32x16` and
  `16x16x32` atoms are **gfx950-only** and are rejected by the spec validator on
  gfx942. The flash-class atom is reached via 2x `iterateK` on `32x32x8`, not by
  widening K per atom.
- **64 KB LDS / CU, 32 banks.** The binding budget for residency.
- **2-CTA/CU threshold = ≤ 32,768 B/CTA.** With 64 KB/CU, two workgroups fit
  only when each uses ≤ 32 KB of LDS. This single number drives the occupancy
  lever below.

---

## Conflict-free operand feed (no transpose-read)

Because gfx942 has no `ds_read_tr_b16`, a transposed-and-conflict-free LDS
operand must be built by transposing on the **store** side, in registers. This
is the pattern CK and flash use:

```text
buffer_load_vN (V in NATURAL [token, dim] layout, vectorized, pipelined HW DMA)
  -> in-register 2x2 transpose with v_perm  (vec_extract / vec_pack;
       __builtin_amdgcn_perm masks 0x01000504 and 0x03020706)
  -> single smem_store of the now-A-consumable layout
  -> plain conflict-free ds_read_b64 in the consumer
```

The transpose is `v_perm` (VALU), so it does **not** touch the LDS port. This is
the only conflict-free transposed-operand path on gfx942.

**Why `v_perm`, not `ds_swizzle` / `warp_shuffle_xor`, for in-register reshapes.**
`ds_swizzle` and `warp_shuffle_xor` (and `ds_bpermute`) are **LDS-port ops**:
they cost `lgkmcnt` and serialize against every other LDS read/write in flight.
On the gfx942 attention path, a register-V transpose built from `ds_swizzle`
measured the `lgkmcnt` wait climbing **18 → 234** and the kernel went flat — the
swizzle serialization cost exactly what the conflict removal saved. CK's
`shuffle_tile@store` dispatches the reshape **in-thread** (arranging the tile
distribution so each lane already owns its head-dim's tokens) for precisely this
reason. Use VALU `v_perm` / `permlane32_swap` for in-register reshapes; reach for
`ds_swizzle` / `ds_bpermute` only for genuine cross-lane traffic with no VALU
equivalent. (The general "`ds_swizzle` is an LDS-port op" note lives in
`primitives/wave_and_cross_lane.md`.)

### The V-feed vehicle taxonomy

When a kernel needs a transposed V operand that is also conflict-free, and the
arch has no transposing async DMA and no `ds_read_tr_b16` (gfx942), there are
three vehicles. Two are traps; record them so nobody re-walks them:

| Vehicle | Correct? | Verdict | Why |
|---|---|---|---|
| (a) sync per-element `global_load` gather (transpose during the HBM read) | yes | ~5x slower | the synchronous VMEM gather stalls on HBM; not pipelined (measured ~30 TF vs L4's 148-178 on D128 fp16) |
| (b) async-DMA into an LDS staging slab, then transpose | yes | LDS-budget wall | extra LDS hop on a 64 KB/CU arch already at the occupancy threshold |
| (c) **register `buffer_load` + in-register `v_perm` transpose + single `smem_store`** | yes | **the win** | the CK / flash way: pipelined HW DMA in natural layout, VALU transpose (no LDS port), plain `ds_read_b64` consumer |

Vehicle (c) is the register-transpose-on-store pattern above. Evidence and the
units-bug that long masked (a)'s correctness:
`architecture/attention_2d_gfx942_experiment_summary.md` (Batch 5).

---

## Occupancy / LDS budget

The whole gfx942 D128 residency game is crossing the **32,768 B/CTA** line to
get from 1 to 2 workgroups per CU.

- **K single-buffer to cross 32,768 B → 2 wg/CU (the L4 win).** The committed
  path drops K from double- to single-buffer, cutting loop LDS 48 → 32 KB, which
  crosses the 2-CTA threshold. Combined with the register-P^T atom path this is
  the shipped lever: **148–178 TF, +18–33% over baseline, ≈62% of flash, 2 wg/CU**
  (commit `9cff43bdd2d`, levers `a07e1f5` + `aa4b118`).
- **acc → AGPR was MOOT for occupancy here.** Moving the accumulator to AGPR to
  free LDS does nothing on this path: the backend's LDS-coalescing already
  aliases `Acc_lds` into the loop-dead K/V region, so there is no LDS to reclaim.

---

## Compile-time budget (KEEP COMPILE FAST)

The gfx942 attention kernel cold-compiles at roughly **250–290 s per
shape-signature** (e.g. bf16-narrow D64 ≈ 245 s; cfv GqaD128 ≈ 293 s). **Treat
this as the budget to respect.** The cold-compile is **per-shape-signature** —
there is no cross-shape cache, so every new shape pays the full cost.

What blows the budget:

- **Fully-unrolled transposes EXPLODE compile time** — a fully-unrolled
  per-element register transpose hit a **45-minute comgr/JIT timeout**. The DSL
  never re-rolls loops, so a Python-time `static_for` / `unroll` over a whole
  tile emits O(tile) straight-line IR.
- **Fix: ALWAYS loop-roll.** Wrap whole-tile reshapes in a runtime `scf_for`
  over micro-tiles and keep only the tiny inner block unrolled; the IR collapses
  and the build drops back to seconds with bit-identical numerics.
- **Prefer native vector ops** (`v_perm`, vector loads/stores) over scalar
  `vec_extract` / `vec_pack` chains — the scalar chains balloon IR for the same
  numerics.

The general (DSL-wide) statement of this principle — IR size ↔ comgr walltime,
and the loop-roll fix — lives in `optimization/optimization_runbook.md` §10.3
and `runtime/comgr_and_hipmodule.md`. The gfx942-attention numbers above are the
concrete budget.

> A compile-time-reduction effort is in progress; link it here when it lands.

---

## Pointers

- **Living case study:** `architecture/attention_2d_gfx942_experiment_summary.md`
  — the full per-batch experiment log (commits, ISA evidence, every kept and
  reverted lever).
- **Empirical pass / results ledger:** `optimization/runbook_compliance.md`
  ("Empirical pass — gfx942 unified attention 2D (CDNA3)") — kept levers,
  proven-negatives, and the conflict-free-V perf placeholder.
