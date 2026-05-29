// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/core/arch/mma/mma_data_format.hpp"
#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/float8.hpp"
#include "ck_tile/core/numeric/pk_fp4.hpp"
#include "ck_tile/core/numeric/pk_f6.hpp"

#include <cstdint>
#include <stdio.h>
#if CK_TILE_CONCEPTS && CK_TILE_CONCEPTS_HEADER
#include <concepts>
#include <type_traits>
#endif // CK_TILE_CONCEPTS && CK_TILE_CONCEPTS_HEADER

namespace ck_tile::core::arch::mma {

struct DefaultScaleMfmaCtrlFlags
{
    using ScaleType                  = int32_t;
    static constexpr int32_t OPSEL_A = 0;
    static constexpr int32_t OPSEL_B = 0;
};

CK_TILE_HOST_DEVICE void print_flags(DefaultScaleMfmaCtrlFlags const& ctrlFlags)
{
    printf("CtrlFlags      OPSEL_A / OPSEL_B        : %d / %d\n",
           ctrlFlags.OPSEL_A,
           ctrlFlags.OPSEL_B);
}

/**
 * @struct DefaultScaleWmmaCtrlFlags
 * @brief Default WMMA scale control flags for GFX1250 scale WMMA operations.
 *
 * The nested @c ScaleType alias selects which underlying builtin family is used:
 *   - @c int32_t -> @c __builtin_amdgcn_wmma_scale_*    (single E8M0 scale per operand)
 *   - @c int64_t -> @c __builtin_amdgcn_wmma_scale16_*  (16 packed E8M0 scales per operand)
 */
struct DefaultScaleWmmaCtrlFlags
{
    using ScaleType = int32_t;
};

CK_TILE_HOST_DEVICE void print_flags(DefaultScaleWmmaCtrlFlags const&)
{
    printf("CtrlFlags      (ScaleWmma, scale-width=32)\n");
}

/**
 * @struct Scale16WmmaCtrlFlags
 * @brief WMMA scale control flags selecting the scale16 builtin family on GFX1250.
 *
 * Identical shape/layout to @ref DefaultScaleWmmaCtrlFlags, but the scale operands are
 * 64-bit values that pack 16 E8M0 sub-scales each, dispatching to the
 * @c __builtin_amdgcn_wmma_scale16_* builtins.
 */
struct Scale16WmmaCtrlFlags
{
    using ScaleType = int64_t;
};

CK_TILE_HOST_DEVICE void print_flags(Scale16WmmaCtrlFlags const&)
{
    printf("CtrlFlags      (ScaleWmma, scale-width=64)\n");
}

#if CK_TILE_CONCEPTS && CK_TILE_CONCEPTS_HEADER

/**
 * @concept ScaleWmmaCtrlFlags
 * @brief Expresses the interface required for scale WMMA control flag types.
 */
template <typename CtrlFlags>
concept ScaleWmmaCtrlFlags = requires {
    typename CtrlFlags::ScaleType;
    requires std::is_integral_v<typename CtrlFlags::ScaleType>;
};

#endif // CK_TILE_CONCEPTS && CK_TILE_CONCEPTS_HEADER

#if CK_TILE_CONCEPTS && CK_TILE_CONCEPTS_HEADER

/**
 * @concept ScaleMfmaCtrlFlags
 * @brief  Expresses the interface of required members for each CtrlFlags type on Gfx9
 */
template <typename CtrlFlags>
concept ScaleMfmaCtrlFlags = requires(CtrlFlags ctrlFlags) {
    // Flag members for scale MFMA instructions
    { CtrlFlags::OPSEL_A } -> std::convertible_to<int32_t>;
    { CtrlFlags::OPSEL_B } -> std::convertible_to<int32_t>;
};

#endif // CK_TILE_CONCEPTS && CK_TILE_CONCEPTS_HEADER

} // namespace ck_tile::core::arch::mma
