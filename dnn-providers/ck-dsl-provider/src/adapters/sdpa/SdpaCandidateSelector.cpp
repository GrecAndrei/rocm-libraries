// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "SdpaCandidateSelector.hpp"

#include <array>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <string>

namespace ck_dsl_provider {

namespace {

using ck_tile::dispatcher::FmhaApiFamily;
using ck_tile::dispatcher::FmhaKernelFamily;
using ck_tile::dispatcher::FmhaKernelKey;
using ck_tile::dispatcher::FmhaProblem;
using ck_tile::dispatcher::FmhaProblemBuilder;

// Enumerated continuous axes. These mirror the kernel's valid sets
// (UnifiedAttention2DTiledSpec.__post_init__): num_warps in {1,2,4,8},
// block_m_per_warp in {16,32}.
constexpr std::array<std::int32_t, 4> kNumWarpsChoices{1, 2, 4, 8};
constexpr std::array<std::int32_t, 2> kBlockMPerWarpChoices{16, 32};

// tile_size is enumerated as multiples of block_size: {1,2,4} * block.
// 2*block_size is P's universal analytic default; the {1,4} multiples
// widen the search while staying within the per-call payload limits the
// supports() gate re-checks.
constexpr std::array<std::int32_t, 3> kTileSizeMultiples{1, 2, 4};

// head_size (= hdim_v) -> k0max for the scored tile_shape, mirroring the
// trained K0_MAX_SUBMAX_MAP. k0max feeds the model's tk0max feature, so a
// faithful value (not the old 0 stub) is required for the score to land in
// the model's trained distribution. Unmapped head sizes fall back to
// head_size (see knobsToKernelKey).
std::uint16_t k0maxForHeadSize(std::int32_t headSize) {
    switch (headSize) {
        case 64:
            return 64;
        case 80:
            return 96;
        case 96:
            return 128;
        case 128:
            return 128;
        case 160:
            return 256;
        case 192:
            return 192;
        case 256:
            return 256;
        default:
            return static_cast<std::uint16_t>(headSize);
    }
}

}  // namespace

SupportsResult supportsTiled2d(const SdpaSelectionProblem& problem, const SdpaPerfKnobs& knobs) {
    // dtype in {fp16, bf16}.
    if (problem.dtype != "fp16" && problem.dtype != "bf16") {
        return {false, "tiled 2D kernel currently supports fp16/bf16 (got " + problem.dtype + ")"};
    }
    // head_size in {64,128,256} and divisible by 32.
    const std::int32_t hd = problem.head_size;
    if (hd != 64 && hd != 128 && hd != 256) {
        return {false, "tiled 2D kernel only supports head_size in {64,128,256} (got " +
                           std::to_string(hd) + ")"};
    }
    if (hd % 32 != 0) {
        return {false, "tiled 2D kernel requires head_size divisible by 32 (got " +
                           std::to_string(hd) + ")"};
    }
    // block_size in {16,32,64}.
    const std::int32_t bs = problem.block_size;
    if (bs != 16 && bs != 32 && bs != 64) {
        return {false, "tiled 2D kernel only supports block_size in {16,32,64} (got " +
                           std::to_string(bs) + ")"};
    }
    // GQA: 1 <= num_queries_per_kv <= 16.
    const std::int32_t qpk = problem.num_queries_per_kv();
    if (qpk < 1 || qpk > 16) {
        return {false, "tiled 2D kernel needs 1<=num_queries_per_kv<=16 (got " +
                           std::to_string(qpk) + ")"};
    }
    // GQA divides the base-row BLOCK_M = 16 * num_warps. NOTE: this is the
    // kernel's base-row form (attention_tiled_2d.py:634), NOT the launch
    // BLOCK_M (num_warps * block_m_per_warp).
    const std::int32_t blockMBase = 16 * knobs.num_warps;
    if (blockMBase % qpk != 0) {
        return {false, "tiled 2D kernel needs num_queries_per_kv to divide BLOCK_M=" +
                           std::to_string(blockMBase) +
                           " (num_warps=" + std::to_string(knobs.num_warps) +
                           ", got num_queries_per_kv=" + std::to_string(qpk) + ")"};
    }
    // tile_size: when set, > 0 and a multiple of block_size.
    if (knobs.tile_size != 0) {
        if (knobs.tile_size <= 0 || knobs.tile_size % bs != 0) {
            return {false, "tiled 2D kernel: tile_size=" + std::to_string(knobs.tile_size) +
                               " must be a positive multiple of block_size=" + std::to_string(bs)};
        }
        // DMA payload floor: tile_size*head_size >= num_warps*64*8.
        const std::int32_t threads = knobs.num_warps * 64;
        if (knobs.tile_size * hd < threads * 8) {
            return {false,
                    "tiled 2D kernel: tile_size*head_size=" + std::to_string(knobs.tile_size * hd) +
                        " too small for num_warps=" + std::to_string(knobs.num_warps) +
                        " (need >= " + std::to_string(threads * 8) + ")"};
        }
        // Per-wave token uniformity: (64*8)/head_size <= block_size.
        const std::int32_t perWaveTokens = (64 * 8) / hd;
        if (perWaveTokens > bs) {
            return {false, "tiled 2D kernel: per-wave tokens " + std::to_string(perWaveTokens) +
                               " exceeds block_size=" + std::to_string(bs) +
                               "; would need lane-divergent block lookup"};
        }
    }
    // use_mfma_32x32 (__post_init__, attention_tiled_2d.py:354-367): requires
    // block_m_per_warp == 32, tile_size_eff % 32 == 0, and head_size % 16 == 0.
    // tile_size_eff falls back to block_size when tile_size is unset. These are
    // __post_init__ constraints (not part of supports_tiled_2d proper) but the
    // builder enforces them too, so the enumerator must respect them to emit
    // only buildable combos.
    if (knobs.use_mfma_32x32) {
        if (knobs.block_m_per_warp != 32) {
            return {false, "tiled 2D kernel: use_mfma_32x32 requires block_m_per_warp=32"};
        }
        const std::int32_t tileEff = knobs.tile_size != 0 ? knobs.tile_size : bs;
        if (tileEff % 32 != 0) {
            return {false, "tiled 2D kernel: use_mfma_32x32 requires tile_size (eff=" +
                               std::to_string(tileEff) + ") to be a multiple of 32"};
        }
        if (hd % 16 != 0) {
            return {false, "tiled 2D kernel: use_mfma_32x32 requires head_size divisible by 16"};
        }
    }
    // LDS-budget gate (AHEAD-OF-TIME compilability). The unified kernel stages
    // its tiles in LDS; comgr CODEGEN (CODEGEN_BC_TO_RELOCATABLE) rejects a
    // kernel whose static group segment exceeds the target arch's LDS capacity
    // (arch_specs.json lds_capacity_bytes: gfx950 163840 B / 160 KB, gfx942
    // 65536 B / 64 KB). This mirrors the kernel's actual smem_alloc footprint
    // (the smem_alloc calls
    // in attention_tiled_2d.py build_unified_attention_2d_tiled; kept in
    // lockstep with that file's supports_tiled_2d LDS gate, bpe=2 fp16/bf16):
    //   K_lds  = 2*T*hd*bpe   (double-buffered async load)
    //   V_lds  =   T*hd*bpe   (single-buffered)
    //   P_lds  = BLOCK_M*(T+8)*bpe  (score tile; +8 pad)
    //   Q_lds  = BLOCK_M*hd*bpe, but 0 when it aliases K (BLOCK_M <= 2*T)
    //   Acc_lds= BLOCK_M*OUT_STRIPE*bpe,  OUT_STRIPE = (hd<=64)?32:hd
    // where T = tile_size_eff, BLOCK_M = num_warps*block_m_per_warp. The model
    // now (after the predict dtype fix) prefers large tiles, so without this
    // gate the heuristic's pick comes back unbuildable. Conservative: assumes
    // P_lds is present (the bf16 REGISTER_PV path drops it) and is LDS-only, so
    // it may reject a couple of register-pressure-borderline mfma32 configs --
    // safe (never emits an unbuildable combo); it keeps the oracle-best
    // tile_size=64 configs. The authoritative copy belongs DSL-side
    // (attention_tiled_2d.py supports_tiled_2d); this is the C++ enumerator
    // mirror, kept in lockstep.
    {
        // Per-arch LDS capacity (arch_specs.json lds_capacity_bytes),
        // mirrored byte-for-byte against the DSL per-arch tiled-2D gate so
        // C++ and Python admit/reject the same configs:
        //   gfx950 -> 163840 B (160 KB)
        //   gfx942 -> 65536 B  (64 KB)
        // Any other arch is never admitted by the applicability gate on
        // this path, so the gfx950 budget is the safe default.
        const std::int64_t kLdsCapacityBytes = (problem.arch == "gfx942") ? 65536 : 163840;
        constexpr std::int64_t kBytesPerElem = 2;  // fp16/bf16
        const std::int64_t tEff = knobs.tile_size != 0 ? knobs.tile_size : bs;
        const std::int64_t blockM =
            static_cast<std::int64_t>(knobs.num_warps) * knobs.block_m_per_warp;
        const std::int64_t outStripe = hd <= 64 ? 32 : hd;

        // gfx942 D128 fp16 plain-mw32: the SHIPPED kernel is the transposed-x8
        // wide-tile L4 (the Python compile path adds use_mfma_32x32x8 +
        // use_transposed_qk_32x32 to this plain candidate). That path's BINDING
        // LDS is much smaller than the generic model above assumes, so the
        // generic count wrongly walls the wide (num_warps=4 / BLOCK_M=128 /
        // tile=64) geometry (100352 B > 64 KB) that measures +19.7%. Mirror the
        // Python ``_lds_bytes_transposed_x8`` accurate model here so the wide
        // candidate survives enumeration and the analytic argmax can pick it:
        //   * P_lds = 0   (P^T is consumed from registers, never staged in LDS).
        //   * K_lds = (1 if BLOCK_M<=T else 2) * T * hd * bpe  (K is single-
        //             buffered while BLOCK_M<=tile; double otherwise -- exactly
        //             the wide chooser's use_k_single_buffer relation).
        //   * V_lds = T * hd * bpe.
        //   * Q_lds = 0 when BLOCK_M <= 2*T (aliases K_lds).
        //   * Acc_lds ~ 0: epilogue-only and the AMDGPU LDS-coalescing pass
        //     aliases it into the loop-dead K/V region (STEP-0 evidence).
        // Gated to gfx942 + D128 + fp16 + plain mw32 (use_mfma_32x32=false): the
        // mfma32 / D64 / bf16 / gfx950 candidates keep the conservative generic
        // model, so their admit/reject set is byte-identical to before.
        const bool gfx942WideL4 = problem.arch == "gfx942" && hd == 128 &&
                                  problem.dtype == "fp16" && knobs.block_m_per_warp == 32 &&
                                  !knobs.use_mfma_32x32;
        std::int64_t ldsBytes;
        if (gfx942WideL4) {
            const std::int64_t kSlots = (blockM <= tEff) ? 1 : 2;
            const std::int64_t kLds = kSlots * tEff * hd * kBytesPerElem;
            const std::int64_t vLds = tEff * hd * kBytesPerElem;
            const std::int64_t qLds = (blockM <= 2 * tEff) ? 0 : blockM * hd * kBytesPerElem;
            ldsBytes = kLds + vLds + qLds;  // P_lds = 0, Acc_lds aliased ~ 0
        } else {
            const std::int64_t kLds = 2 * tEff * hd * kBytesPerElem;
            const std::int64_t vLds = tEff * hd * kBytesPerElem;
            const std::int64_t pLds = blockM * (tEff + 8) * kBytesPerElem;
            const std::int64_t qLds = (blockM <= 2 * tEff) ? 0 : blockM * hd * kBytesPerElem;
            const std::int64_t accLds = blockM * outStripe * kBytesPerElem;
            ldsBytes = kLds + vLds + pLds + qLds + accLds;
        }
        if (ldsBytes > kLdsCapacityBytes) {
            return {false, "tiled 2D kernel: estimated LDS " + std::to_string(ldsBytes) +
                               " B exceeds the " + problem.arch + " " +
                               std::to_string(kLdsCapacityBytes) +
                               " B budget; comgr CODEGEN would fail"};
        }
    }
    return {true, ""};
}

std::vector<SdpaPerfKnobs> enumerateCandidates(const SdpaSelectionProblem& problem) {
    std::vector<SdpaPerfKnobs> out;

    for (const std::int32_t numWarps : kNumWarpsChoices) {
        for (const std::int32_t blockMPerWarp : kBlockMPerWarpChoices) {
            // block_m_per_warp == 32 requires num_warps in {1,2,4}
            // (CTA thread cap).
            if (blockMPerWarp == 32 && numWarps == 8) {
                continue;
            }
            for (const std::int32_t mult : kTileSizeMultiples) {
                const std::int32_t tileSize = mult * problem.block_size;

                // Curated atom/schedule flag set. The 16x16x32 atom is
                // the default; the 32x32x16 atom requires
                // block_m_per_warp == 32. We enumerate the atom choice
                // and (for the default atom) the register-PV /
                // early-V schedule variants. The OI-B default-fill
                // table (see below) sets every non-enumerated flag.
                std::vector<SdpaPerfKnobs> flagVariants;

                // Default 16x16x32 atom, plain schedule.
                {
                    SdpaPerfKnobs k;
                    k.num_warps = numWarps;
                    k.block_m_per_warp = blockMPerWarp;
                    k.tile_size = tileSize;
                    flagVariants.push_back(k);
                }
                // Default atom + early-V schedule (a pipeline-affecting
                // schedule flag, valid on the 16x16x32 path).
                {
                    SdpaPerfKnobs k;
                    k.num_warps = numWarps;
                    k.block_m_per_warp = blockMPerWarp;
                    k.tile_size = tileSize;
                    k.use_early_v_schedule = true;
                    flagVariants.push_back(k);
                }
                // 32x32x16 atom (only when block_m_per_warp == 32).
                if (blockMPerWarp == 32) {
                    SdpaPerfKnobs k;
                    k.num_warps = numWarps;
                    k.block_m_per_warp = blockMPerWarp;
                    k.tile_size = tileSize;
                    k.use_mfma_32x32 = true;
                    k.use_transposed_qk_32x32 = true;
                    flagVariants.push_back(k);
                }

                for (SdpaPerfKnobs k : flagVariants) {
                    // OI-B default-fill: carry the problem-driven
                    // variant lanes so every emitted combo is complete.
                    // The remaining boolean flags keep their POD
                    // defaults (all false), which is supports()-valid.
                    k.use_sinks = problem.use_sinks;
                    k.sliding_window = problem.sliding_window;
                    k.waves_per_eu = 0;  // unset -> LLVM heuristic

                    if (supportsTiled2d(problem, k).supported) {
                        out.push_back(k);
                    }
                }
            }
        }
    }
    return out;
}

std::string pipelineForKnobs(const SdpaSelectionProblem& problem, const SdpaPerfKnobs& knobs) {
    // SCORING pipeline token. The fwd model (gfx950) was trained ONLY on
    // the "qr" / "qr_async" pipelines; "qr_pagedkv" and the "qr_async_trload*"
    // tokens are out-of-vocabulary and pull the prediction off-distribution.
    // We score every current candidate as "qr_async": it is the codegen
    // default, dominates the gfx950 fwd training set, and matches our
    // kernel's async K/V DMA. (We do NOT score the serial "qr": the tiled 2D
    // kernel always issues async DMA.) This is a scoring-only token; it does
    // not change the compiled kernel or the enumerated knob set.
    (void)problem;
    (void)knobs;
    return "qr_async";
}

FmhaProblem problemToFmhaProblem(const SdpaSelectionProblem& problem) {
    return FmhaProblemBuilder()
        .api_family(FmhaApiFamily::Fwd)
        .kernel_family(FmhaKernelFamily::Fwd)
        .gfx_arch("gfx950")
        .data_type(problem.dtype)
        .dims(/*hdim_q=*/problem.head_size,
              /*hdim_v=*/problem.head_size,
              /*batch=*/problem.batch,
              /*seqlen_q=*/problem.seqlen_q,
              /*seqlen_k=*/problem.seqlen_k)
        .nheads(/*q=*/problem.num_query_heads, /*k=*/problem.num_kv_heads)
        .mask_type(problem.mask_type)
        .bias_type(problem.bias_type)
        .paged_kv(problem.use_paged_kv)
        .sink(problem.use_sinks)
        .skip_min_seqlen_q(problem.skip_min_seqlen_q)
        .build();
}

FmhaKernelKey knobsToKernelKey(const SdpaSelectionProblem& problem, const SdpaPerfKnobs& knobs) {
    FmhaKernelKey key;

    // --- Signature: matched to the problem (scored flags) ------------
    key.signature.family = FmhaKernelFamily::Fwd;
    key.signature.data_type = problem.dtype;
    key.signature.mask_type = problem.mask_type;
    key.signature.bias_type = problem.bias_type;
    key.signature.has_lse = false;  // paged kernel has no LSE output
    key.signature.has_dropout = false;
    key.signature.has_logits_soft_cap = false;  // not expressible from the graph
    key.signature.has_sink = problem.use_sinks;
    key.signature.skip_min_seqlen_q = problem.skip_min_seqlen_q;
    // SCORING-only: the fwd model never saw paged candidates. A paged=1
    // query inflates the dominant feature_count split and zeroes the
    // ratio_dv_to_n1 feature, collapsing every shape onto the smallest
    // config. We score as dense (use_paged_kv=false); this is the
    // prediction query only and does NOT touch the runtime marshalling
    // path (SdpaFwdPlanBuilder keeps selProblem.use_paged_kv = true).
    key.signature.use_paged_kv = false;
    key.signature.hdim_q = static_cast<std::uint16_t>(problem.head_size);
    key.signature.hdim_v = static_cast<std::uint16_t>(problem.head_size);

    // --- Algorithm.tile_shape: the scored continuous axes ------------
    // m0 = launch BLOCK_M = num_warps * block_m_per_warp (NOT the GQA
    // base-row 16*num_warps used by supportsTiled2d).
    key.algorithm.tile_shape.m0 = static_cast<std::uint16_t>(knobs.block_m());
    key.algorithm.tile_shape.n0 = static_cast<std::uint16_t>(knobs.tile_size);
    // Trained-faithful block-K fields. The trained convention pairs the
    // MFMA K-atom wk0 (=32 for the default 16x16x32 atom, =16 for the
    // 32x32x16 use_mfma_32x32 atom) with a block-K bk0 ("tile_k0") near 32
    // that satisfies bk0 % wk0 == 0 and bk0 <= D, plus bk1 ("tile_k1") = 32.
    // bk0 = bk1 = 32 satisfies 32 % wk0 == 0 for wk0 in {16,32} and 32 <= D
    // for D in {64,128,256}, so both atoms share k0 = k1 = 32 here.
    key.algorithm.tile_shape.k0 = 32;
    key.algorithm.tile_shape.k1 = 32;
    // bn1 ("tile_n1") = hdim_v (= head_size); k0max from the trained
    // head_size -> k0max map. These feed the model's tn1 / tk0max features;
    // the old 0 stubs zeroed ratio_dv_to_n1 and the k0max split.
    key.algorithm.tile_shape.n1 = static_cast<std::uint16_t>(problem.head_size);
    key.algorithm.tile_shape.k0max = k0maxForHeadSize(problem.head_size);

    // --- Pipeline string: trained-vocabulary token ("qr_async") ------
    key.algorithm.pipeline = pipelineForKnobs(problem, knobs);

    // --- pad_*: no CK DSL analog -> default-filled (true) ------------
    key.algorithm.pad_s = true;
    key.algorithm.pad_sk = true;
    key.algorithm.pad_d = true;
    key.algorithm.pad_dv = true;

    // --- Non-scored fields: sensible supports()-valid defaults -------
    // wave_shape / warp_tile_shape / alignments / block_per_cu /
    // selection_rank are NOT read by predict_tflops; they keep the
    // struct's defaults except block_per_cu, which we map from
    // num_warps so the identifier stays distinct per combo.
    key.algorithm.block_per_cu = static_cast<std::uint8_t>(knobs.num_warps);

    key.gfx_arch = "gfx950";
    return key;
}

SdpaPerfKnobs selectArgmax(
    const SdpaSelectionProblem& problem, const std::vector<SdpaPerfKnobs>& candidates,
    const std::function<double(const ck_tile::dispatcher::FmhaKernelKey&)>& score) {
    // Caller guarantees non-empty.
    std::size_t bestIdx = 0;
    double bestScore = -std::numeric_limits<double>::infinity();

    for (std::size_t i = 0; i < candidates.size(); ++i) {
        const FmhaKernelKey key = knobsToKernelKey(problem, candidates[i]);
        const double s = score(key);
        // Strict > preserves enumeration order on ties (the first combo
        // with the maximal score wins) -> stable, deterministic.
        if (i == 0 || s > bestScore) {
            bestScore = s;
            bestIdx = i;
        } else if (s == bestScore && candidates[i].use_mfma_32x32 &&
                   !candidates[bestIdx].use_mfma_32x32) {
            // MFMA-atom tie-break. The scoring key is byte-identical for the
            // mfma32 and non-mfma32 variants of the same (m0, n0, tile) -- the
            // model has no warp-atom feature -- so predict() returns the exact
            // same score and they tie. The 32x32x16 atom is the oracle-best on
            // the D128 shapes the model targets, so break the tie toward it.
            bestIdx = i;
        }
    }
    return candidates[bestIdx];
}

namespace {

// Analytic target geometry derived from P's production policy
// (_select_2d_num_warps / _select_2d_tile_size / _select_2d_block_m_per_warp
// in attention_unified.py). For the Phase-1 scope (no fp8, no combo/
// transposed path) the policy reduces to:
//
//   * tile_size  = 2 * block_size            (universal default)
//                  block_size                (when sliding_window > 0)
//   * num_warps  by max_seqlen_q (~ seqlen_q here):
//                  <= 64  -> 1
//                  <= 128 -> 2
//                  <= 256 -> 4
//                  > 256  -> 2   (single-seq long prefill; POC treats
//                                 each shape as one logical sequence)
//   * block_m_per_warp = 16                  (mw=32 only on fp8 / combo
//                                             paths, out of Phase-1 scope)
//   * all schedule flags off                 (plain async pipeline)
//
// The architectural clamps (DMA floor, per-wave tokens) are applied by
// re-validating against supportsTiled2d.
struct AnalyticTarget {
    std::int32_t num_warps;
    std::int32_t block_m_per_warp;
    std::int32_t tile_size;
    // When true, the early-V schedule is rewarded (not penalised) by
    // analyticCloseness so the analytic pick lands on the early-V variant.
    bool prefer_early_v = false;
};

AnalyticTarget analyticTarget(const SdpaSelectionProblem& problem) {
    const std::int32_t bs = problem.block_size;

    // gfx942 (CDNA3 / MI300X) head_size=64 family. The general policy below
    // picks mw=16 + 2x-tile, but on gfx942 the D64 oracle is always mw=32
    // with a 1x tile (tile_size == block_size). gfx942 is LDS-bound at one
    // CTA/CU on the 64 KB part: the mw=32 + 2x-tile combo (e.g.
    // nw4/mw32/t128 ~= 70 KB) is LDS-rejected by supportsTiled2d, while the
    // 1x-tile mw=32 combo (nw4/mw32/t64 ~= 51 KB) fits. Oracle data on this
    // MI300X shows 1.7-2.0x over the mw=16/2x-tile pick. num_warps follows
    // the oracle by tile granularity: the larger block (bs>=64 -> T=64 KV
    // tokens/iter) amortizes BLOCK_M=128 (nw4), while the smaller block
    // (bs=32 -> T=32) prefers BLOCK_M=64 (nw2). The shared DMA-floor clamp
    // then applies.
    if (problem.arch == "gfx942" && problem.head_size == 64) {
        const std::int32_t tileSize = bs;  // 1x multiple so mw=32 fits 64 KB LDS
        std::int32_t numWarps = bs >= 64 ? 4 : 2;
        while (numWarps > 1 && tileSize * problem.head_size < numWarps * 64 * 8) {
            numWarps /= 2;
        }
        return {numWarps, /*block_m_per_warp=*/32, tileSize};
    }

    // gfx942 (CDNA3 / MI300X) head_size=128 fp16: SHIP the validated wide-tile L4
    // kernel (transposed-x8, WG=256 / num_warps=4). Sub-lever 3 of the
    // flash-regime rewrite measured +19.7% over the WG=64 L4 baseline (153.6 ->
    // 183.8 TF, 53% -> 63% of PyTorch flash; job 354653), correct 16/2/0 (job
    // 354658), gfx950 byte-identical. rocprof (job 354659) pinned the win as the
    // latency-hider the WG=64 kernel lacked: 4 resident waves/WG raise wave-slot
    // fill 4.59% -> 7.59% so the LDS latency that starved MFMA at WG=64 is now
    // overlapped. The Python compile path (_unified_tiled_spec_from_problem)
    // adds the L4 knobs that do NOT travel on the wire -- ``use_mfma_32x32x8`` +
    // ``use_transposed_qk_32x32`` (K is double-buffered at this width since
    // BLOCK_M=128 > tile=64, so ``use_k_single_buffer`` is dropped by the wide
    // chooser) -- for every gfx942 D128 fp16 build by default; the C++ selector's
    // only job is to pick the matching GEOMETRY so the launch grid AND the folded
    // cache key (GraphSignature folds knobs.num_warps) stay coherent with the
    // wide kernel.
    //
    // num_warps=4 / block_m_per_warp=32 (BLOCK_M=128) / tile_size=64 mirrors the
    // wide chooser's pick exactly. We return the PLAIN flags (use_mfma_32x32=
    // false): the wire's ``use_mfma_32x32`` is the gfx950 32x32x16 atom and would
    // BLOCK the Python wide/L4 branch (it gates on ``not use_mfma_32x32``), so
    // the analytic pick must be the plain nw4/mw32/t64 candidate.
    // ``analyticCloseness`` penalises the mfma/transposed-bundled candidate (+8)
    // and minimises |num_warps - 4|, so the plain nw4 combo wins the argmax. This
    // plain candidate survives enumeration because supportsTiled2d now scores the
    // gfx942 D128 fp16 plain-mw32 path with the ACCURATE transposed-x8 LDS model
    // (P_lds=0 / Acc_lds aliased), which the wide kernel actually realises. The
    // Python ``HIPDNN_GFX942_FLASH_WIDE=0`` kill-switch can still revert the BUILT
    // kernel to WG=64 (a debug/revert escape hatch, same class as
    // HIPDNN_GFX942_FLASH_PIPELINE); the production default is fully coherent.
    // bf16 D128 (32x32x8 is fp16-only) and all D64 are left to the branches
    // above/below -- unchanged. cfv is never selected here.
    if (problem.arch == "gfx942" && problem.head_size == 128 && problem.dtype == "fp16") {
        return {/*num_warps=*/4, /*block_m_per_warp=*/32, /*tile_size=*/64};
    }

    std::int32_t tileSize = 2 * bs;
    if (problem.sliding_window > 0) {
        tileSize = bs;
    }

    std::int32_t numWarps = 4;
    const std::int32_t sq = problem.seqlen_q;
    if (sq <= 64) {
        numWarps = 1;
    } else if (sq <= 128) {
        numWarps = 2;
    } else if (sq <= 256) {
        numWarps = 4;
    } else {
        numWarps = 2;
    }

    // Step num_warps down until the DMA floor is satisfied for this
    // tile_size/head_size, mirroring the kernel's own clamp loop.
    while (numWarps > 1 && tileSize * problem.head_size < numWarps * 64 * 8) {
        numWarps /= 2;
    }

    // gfx942 (CDNA3 / MI300X) head_size=128 GQA: the oracle-best overlaps the
    // V load with QK+softmax (early-V schedule) for ~2-4% over the plain async
    // pipeline, at no LDS cost. Two carve-outs from oracle data on this MI300X:
    // MHA (num_queries_per_kv==1) does not benefit (its oracle-best keeps the
    // plain schedule), and the benefit inverts at very long prefill (seqlen_q
    // 8192 prefers plain — the V load is already well-hidden), so gate to GQA
    // with seqlen_q <= 4096. The continuous axes (nw2/mw16/t64) already match
    // the oracle; only the schedule flag moves.
    const bool preferEarlyV = problem.arch == "gfx942" && problem.head_size == 128 &&
                              problem.num_queries_per_kv() > 1 && problem.seqlen_q <= 4096;

    return {numWarps, /*block_m_per_warp=*/16, tileSize, preferEarlyV};
}

// Distance score: lower is closer to the analytic target. Returned as a
// negated value so the caller can take an argmax consistently with
// selectArgmax's "higher is better" convention.
double analyticCloseness(const SdpaPerfKnobs& cand, const AnalyticTarget& target) {
    double dist = 0.0;
    // num_warps and tile_size are the dominant production axes; weight
    // them heavily. block_m_per_warp is a smaller correction.
    dist += 4.0 * static_cast<double>(std::abs(cand.num_warps - target.num_warps));
    dist += 1.0 * static_cast<double>(std::abs(cand.tile_size - target.tile_size)) /
            static_cast<double>(target.tile_size > 0 ? target.tile_size : 1);
    dist += 2.0 * static_cast<double>(std::abs(cand.block_m_per_warp - target.block_m_per_warp));
    // The production policy uses the plain async pipeline (no atom /
    // schedule variants) for this scope; penalise variant flags so the
    // plain combo wins on a tie of the continuous axes.
    if (cand.use_mfma_32x32 || cand.use_transposed_qk_32x32) {
        dist += 8.0;
    }
    if (cand.use_early_v_schedule) {
        // Reward early-V where the target prefers it (gfx942 D128 GQA), else
        // penalise so the plain async pipeline wins on a tie of the axes.
        dist += target.prefer_early_v ? -0.5 : 0.5;
    }
    return -dist;
}

}  // namespace

SdpaPerfKnobs selectAnalyticFallback(const SdpaSelectionProblem& problem,
                                     const std::vector<SdpaPerfKnobs>& candidates) {
    // Caller guarantees non-empty.
    const AnalyticTarget target = analyticTarget(problem);

    std::size_t bestIdx = 0;
    double bestScore = -std::numeric_limits<double>::infinity();

    for (std::size_t i = 0; i < candidates.size(); ++i) {
        const double s = analyticCloseness(candidates[i], target);
        // Strict > -> first (enumeration-order) candidate wins ties.
        if (i == 0 || s > bestScore) {
            bestScore = s;
            bestIdx = i;
        }
    }
    return candidates[bestIdx];
}

SdpaPerfKnobs selectPerfKnobs(const SdpaSelectionProblem& problem,
                              const std::vector<SdpaPerfKnobs>& candidates,
                              const SdpaScorer& scorer) {
    // Model-load failure (or no model in the tree) degrades to the
    // analytic production policy over the SAME candidate set -- never a
    // trivial first-fit. The fwd lgbm model is gfx950-trained, so any
    // other arch (gfx942) also takes the analytic path: its scorer key
    // would be out-of-distribution. This short-circuits BEFORE any
    // scorer-key construction (problemToFmhaProblem / knobsToKernelKey),
    // so the gfx950-hardcoded gfx_arch in those mappings is dead on the
    // gfx942 path.
    if (!scorer.isLoaded() || problem.arch != "gfx950") {
        return selectAnalyticFallback(problem, candidates);
    }

    // Build the dispatcher FmhaProblem ONCE: signature.* is identical
    // across every scored candidate (only the algorithm.* tile/pipeline
    // fields vary per knob combo via knobsToKernelKey), so re-deriving it
    // inside the scoring lambda would be pure waste.
    const FmhaProblem fmhaProblem = problemToFmhaProblem(problem);

    return selectArgmax(problem, candidates,
                        [&scorer, &fmhaProblem](const FmhaKernelKey& key) -> double {
                            return scorer.predict(fmhaProblem, key);
                        });
}

}  // namespace ck_dsl_provider
