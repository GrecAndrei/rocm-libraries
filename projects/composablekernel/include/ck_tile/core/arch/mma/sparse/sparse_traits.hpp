// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/integer.hpp"

#include <cinttypes>
#include <stdio.h>
#if CK_TILE_CONCEPTS && CK_TILE_CONCEPTS_HEADER
#include <concepts>
#endif

namespace ck_tile::core::arch::mma {

/**
 * @enum SparseCompressionIndex
 * @brief Indicates which set of sparse-indices within a VGPR starting at srcC
 * containing 8-bits (for 16-bit source data) or 16-bits (for 8-bit source data)
 * of index information for a lane. \see DefaultSparseMfmaCtrlFlags
 */
enum struct SparseCompressionIndex : int32_t
{
    FIRST  = 0, // Uses bits  [7:0] or [15..0], for 16 and 8 bit data respectively
    SECOND = 1, // Uses bits [15:8] or [31:16], for 16 and 8 bit data respectively
    THIRD  = 2, // Uses bits [23:16]
    FOURTH = 3, // Uses bits [31:24]
};

// to_string methods for enum classes
CK_TILE_HOST_DEVICE constexpr const char* to_string(SparseCompressionIndex compressionIndex)
{
    switch(compressionIndex)
    {
    case SparseCompressionIndex::FIRST: return "FIRST";
    case SparseCompressionIndex::SECOND: return "SECOND";
    case SparseCompressionIndex::THIRD: return "THIRD";
    case SparseCompressionIndex::FOURTH: return "FOURTH";
    }
    __builtin_unreachable();
}

namespace sparse::detail {

/**
 * @class BuiltinParams
 * @brief Translates the SparseCompressionIndex to the correct CBSZ and ABID pairs for sparse
 * builtins. The actual behavior of the builtin depends on the input data type: 16-bit source data:
 * If CBSZ=0, ABID selects one of four 8-bit sets of sparse-indices within a VGPR starting at srcC
 * containing 8-bits of index information for a lane. If CBSZ!=0 the very first is selected
 * (VGPR[srcC][7..0]).
 *
 * 8-bit source data:
 * If CBSZ=0, ABID selects one of two 16-bit sets of sparse-indices within a VGPR starting at srcC
 * containing 16-bits of index information for a lane. If CBSZ!=0; the very first is selected
 * (VGPR[srcC][15..0]).
 */
template <SparseCompressionIndex Idx>
struct BuiltinParams
{
    static constexpr int32_t UseFirstIndex       = (Idx == SparseCompressionIndex::FIRST); // CBSZ
    static constexpr int32_t ByteIndexToOverride = static_cast<int32_t>(Idx);              // ABID
};

} // namespace sparse::detail

/**
 * @struct DefaultSparseMfmaCtrlFlags
 * @brief Default MFMA sparse flags, select (VGPR[srcC][7..0]) if srcC is
 * 16-bit or (VGPR[srcC][15..0]) if srcC is 8-bit.
 */
struct DefaultSparseMfmaCtrlFlags
{
    static constexpr SparseCompressionIndex CompressionIndex = SparseCompressionIndex::FIRST;
};

CK_TILE_HOST_DEVICE void print_flags(DefaultSparseMfmaCtrlFlags const& ctrlFlags)
{
    using PARAMS = sparse::detail::BuiltinParams<DefaultSparseMfmaCtrlFlags::CompressionIndex>;

    printf("CtrlFlags      CompressionIndex         : %s\n", to_string(ctrlFlags.CompressionIndex));
    printf("               UseFirstIndex            : %" PRId32 "\n", PARAMS::UseFirstIndex);
    printf("               ByteIndexToOverride      : %" PRId32 "\n", PARAMS::ByteIndexToOverride);
}

#if CK_TILE_CONCEPTS && CK_TILE_CONCEPTS_HEADER
/**
 * @concept SparseMfmaCtrlFlags
 * @brief Expresses the interface of required members for each CtrlFlags type
 */
template <typename CtrlFlags>
concept SparseMfmaCtrlFlags = requires(CtrlFlags ctrlFlags) {
    // Flag members for sparse MFMA instructions
    { CtrlFlags::CompressionIndex } -> std::convertible_to<SparseCompressionIndex>;
};
#endif // CK_TILE_CONCEPTS && CK_TILE_CONCEPTS_HEADER

} // namespace ck_tile::core::arch::mma
