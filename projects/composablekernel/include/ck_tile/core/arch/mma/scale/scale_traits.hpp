// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/core/arch/mma/mma_data_format.hpp"
#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/e4m3.hpp"
#include "ck_tile/core/numeric/e5m3.hpp"
#include "ck_tile/core/numeric/e8m0.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/core/numeric/pk_fp4.hpp"

#include <cinttypes>
#include <stdio.h>
#include <type_traits>
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

namespace scale::detail {

struct ScaleWmmaCtrlFlagsBase
{
    static constexpr int16_t c_mod           = 0; // 0: none, 1: neg, 2: abs, 3: neg(abs)
    static constexpr int32_t scaleA_selector = 0;
    static constexpr int32_t scaleB_selector = 0;
};

} // namespace scale::detail

CK_TILE_HOST_DEVICE void print_flags(scale::detail::ScaleWmmaCtrlFlagsBase const& ctrlFlags)
{
    printf("CtrlFlags      c_mod                    : %" PRId16 "\n", ctrlFlags.c_mod);
    printf("               scaleA_selector          : %" PRId32 "\n", ctrlFlags.scaleA_selector);
    printf("               scaleB_selector          : %" PRId32 "\n", ctrlFlags.scaleB_selector);
}

// Default Scale control flags
struct DefaultScaleWmmaCtrlFlags : scale::detail::ScaleWmmaCtrlFlagsBase
{
};

// Scale control flags for GFX1250 scale16 WMMA instructions
struct Scale16WmmaCtrlFlags : scale::detail::ScaleWmmaCtrlFlagsBase
{
};

#if CK_TILE_CONCEPTS && CK_TILE_CONCEPTS_HEADER

/**
 * @concept ScaleWmmaCtrlFlags
 * @brief Expresses the interface required for scale WMMA control flag types.
 */
template <typename CtrlFlags>
concept ScaleWmmaCtrlFlags = requires(CtrlFlags ctrlFlags) {
    // Flag members for scale WMMA instructions
    { CtrlFlags::c_mod } -> std::convertible_to<int16_t>;
    { CtrlFlags::scaleA_selector } -> std::convertible_to<int32_t>;
    { CtrlFlags::scaleB_selector } -> std::convertible_to<int32_t>;
};

#endif // CK_TILE_CONCEPTS && CK_TILE_CONCEPTS_HEADER

namespace scale::detail {

template <typename T>
struct ScaleTypeToFlag;

template <>
struct ScaleTypeToFlag<e8m0_t>
{
    static constexpr int32_t value = 0;
};

template <>
struct ScaleTypeToFlag<e5m3_t>
{
    static constexpr int32_t value = 1;
};

template <>
struct ScaleTypeToFlag<e4m3_t>
{
    static constexpr int32_t value = 2;
};

template <typename T>
inline constexpr int32_t ScaleTypeToFlag_v = ScaleTypeToFlag<T>::value;

template <typename DataType, typename ScaleVecType, bool IsScale16, typename = void>
inline constexpr bool is_valid_ScaleVecType = false;

template <typename ScaleVecType>
inline constexpr bool is_valid_ScaleVecType<pk_fp4_t, ScaleVecType, true, void> =
    std::is_same_v<ScaleVecType, e8m0x8_t> || std::is_same_v<ScaleVecType, e5m3x8_t> ||
    std::is_same_v<ScaleVecType, e4m3x8_t>;

template <typename ScaleVecType>
inline constexpr bool is_valid_ScaleVecType<pk_fp4_t, ScaleVecType, false, void> =
    std::is_same_v<ScaleVecType, e8m0x4_t> || std::is_same_v<ScaleVecType, e5m3x4_t> ||
    std::is_same_v<ScaleVecType, e4m3x4_t>;

template <typename DataType, typename ScaleVecType>
inline constexpr bool
    is_valid_ScaleVecType<DataType,
                          ScaleVecType,
                          true,
                          std::void_t<decltype(PackedDataTypeToFlag<DataType>::value)>> =
        std::is_same_v<ScaleVecType, e8m0x8_t>;

template <typename DataType, typename ScaleVecType>
inline constexpr bool
    is_valid_ScaleVecType<DataType,
                          ScaleVecType,
                          false,
                          std::void_t<decltype(PackedDataTypeToFlag<DataType>::value)>> =
        std::is_same_v<ScaleVecType, e8m0x4_t>;

template <typename ADataType,
          typename BDataType,
          typename ScaleAVecType,
          typename ScaleBVecType,
          bool IsScale16>
inline constexpr bool is_legal_combination =
    is_valid_ScaleVecType<ADataType, ScaleAVecType, IsScale16> &&
    is_valid_ScaleVecType<BDataType, ScaleBVecType, IsScale16> &&
    (!(std::is_same_v<ADataType, pk_fp4_t> && std::is_same_v<BDataType, pk_fp4_t>) ||
     std::is_same_v<ScaleAVecType, ScaleBVecType>);

} // namespace scale::detail

} // namespace ck_tile::core::arch::mma
