// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/integer.hpp"

#include <stdio.h>
#if CK_TILE_CONCEPTS && CK_TILE_CONCEPTS_HEADER
#include <concepts>
#endif // CK_TILE_CONCEPTS && CK_TILE_CONCEPTS_HEADER

namespace ck_tile::core::arch::mma {

struct DefaultScaleMfmaCtrlFlags
{
    static constexpr int32_t OPSEL_A = 0;
    static constexpr int32_t OPSEL_B = 0;
};

CK_TILE_HOST_DEVICE void print_flags(DefaultScaleMfmaCtrlFlags const& ctrlFlags)
{
    printf("CtrlFlags      OPSEL_A / OPSEL_B        : %d / %d\n",
           ctrlFlags.OPSEL_A,
           ctrlFlags.OPSEL_B);
}

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

// Default Scale control flags
struct DefaultScaleWmmaCtrlFlags
{
};

CK_TILE_HOST_DEVICE void print_flags(DefaultScaleWmmaCtrlFlags const&)
{
    printf("CtrlFlags      (ScaleWmma, scale-width=32)\n");
}

// Scale control flags for GFX1250 scale16 WMMA instructions
struct Scale16WmmaCtrlFlags
{
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
concept ScaleWmmaCtrlFlags = requires(CtrlFlags ctrlFlags) { typename CtrlFlags; };

#endif // CK_TILE_CONCEPTS && CK_TILE_CONCEPTS_HEADER

} // namespace ck_tile::core::arch::mma
