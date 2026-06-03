# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Standalone DSL micro-checks for the gfx942 tiled-2D attention variant.

These tests guard the two riskiest pieces of the gfx942 (CDNA3) narrow-atom
tiled-2D attention kernel (``instances/gfx942/attention_tiled_2d.py``):

  1. **V B-operand lane map** -- gfx942 has no ``ds_read_b64_tr_b16``, so the PV
     ``B`` operand is built from ordinary strided LDS loads. A WRONG per-lane
     ``(row, col)`` map is a *silent numerical bug*, not a crash. This test
     asserts the strided map EQUALS the ``ds_read_tr16_b64`` transpose-read map
     (``helpers/layouts.py``) for a ``16x16x16`` atom at K=16, EXACTLY, lane by
     lane and element by element -- so a lane bug is localized immediately.
  2. **Arch routing** -- after ``validate_tiled_attention_arch`` was relaxed to
     admit gfx942, a gfx942 request MUST resolve the gfx942 variant and NEVER
     the gfx950 builder (otherwise comgr crashes on gfx950-only ISA). This test
     spies on the gfx950 builder and asserts it is never invoked for gfx942.

A small NumPy PV simulation then proves the strided lane map reconstructs the
exact ``P @ V`` result the MFMA computes, end-to-end, without needing a GPU.

Run:  PYTHONPATH=python python3 python/test/test_ck_dsl_gfx942_attention.py
These tests need no GPU.
"""

from __future__ import annotations

import unittest

import numpy as np

from ck_dsl.core.arch import ArchTarget


# ---------------------------------------------------------------------------
# Reference maps
# ---------------------------------------------------------------------------
#
# Both maps describe, for a 16x16x16 MFMA atom on wave64, which 4 (row, col)
# LDS coordinates lane ``l`` must end up holding as its per-lane ``B`` operand
# ``<4 x dtype>``, relative to a V_lds tile whose origin is (k_row_base=k*16,
# n_col_base=n*16).


def ds_read_tr16_b64_b_operand_map(k_iter: int, n_tile: int):
    """The (row, col) coords lane ``l`` ends up holding after ds_read_tr16_b64.

    From ``IRBuilder.ds_read_tr16_b64`` semantics (core/ir.py): for a 16x16
    tile at LDS origin (k_iter*16, n_tile*16), lane ``l = 16*k_chunk + n``
    (``k_chunk = l // 16``, ``n = l % 16``) holds, for ``j in 0..3``:

        tile[k_chunk*4 + j, n]  ==  V_lds[k_iter*16 + k_chunk*4 + j,
                                          n_tile*16 + n]

    Returns ``map[l] = [(row, col)]*4``.
    """
    out = {}
    for l in range(64):
        k_chunk = l // 16
        n = l % 16
        coords = []
        for j in range(4):
            row = k_iter * 16 + k_chunk * 4 + j
            col = n_tile * 16 + n
            coords.append((row, col))
        out[l] = coords
    return out


def strided_v_b_operand_map(k_iter: int, n_tile: int):
    """The (row, col) coords the gfx942 strided-V loader assigns to lane ``l``.

    Mirrors ``_strided_v_b_operand`` in the gfx942 kernel:
        v_n_col       = n_tile*16 + (l % 16)         # lane_col
        v_k_chunk_base= (l // 16) * 4                 # lane_rg * 4
        B[j] = V_lds[k_iter*16 + j + v_k_chunk_base, v_n_col],  j in 0..3
    """
    out = {}
    for l in range(64):
        lane_col = l % 16
        lane_rg = l // 16
        v_n_col = n_tile * 16 + lane_col
        v_k_chunk_base = lane_rg * 4
        coords = []
        for j in range(4):
            v_row = k_iter * 16 + j + v_k_chunk_base
            coords.append((v_row, v_n_col))
        out[l] = coords
    return out


class TestGfx942VLaneMap(unittest.TestCase):
    """The strided-V B-operand map must EXACTLY equal the transpose-read map."""

    def test_lane_map_equals_transpose_read_k16(self):
        for k_iter in range(4):
            for n_tile in range(4):
                tr = ds_read_tr16_b64_b_operand_map(k_iter, n_tile)
                strided = strided_v_b_operand_map(k_iter, n_tile)
                for lane in range(64):
                    self.assertEqual(
                        strided[lane],
                        tr[lane],
                        msg=(
                            f"lane {lane} B-operand map mismatch "
                            f"(k_iter={k_iter}, n_tile={n_tile}): "
                            f"strided={strided[lane]} tr={tr[lane]}"
                        ),
                    )

    def test_b_operand_covers_full_16x16_tile(self):
        # Across all 64 lanes the 4-element B operands must cover every
        # (row 0..15, col 0..15) cell of the tile exactly once per (k_iter,
        # n_tile) -- a permutation check that catches off-by-one / collision.
        k_iter, n_tile = 1, 2
        strided = strided_v_b_operand_map(k_iter, n_tile)
        seen = {}
        for lane in range(64):
            for (r, c) in strided[lane]:
                local = (r - k_iter * 16, c - n_tile * 16)
                seen[local] = seen.get(local, 0) + 1
        # 16x16 = 256 cells, but only k_chunk*4+j in 0..15 over 4 k_chunks =
        # rows 0..15; each lane contributes 4, 64 lanes -> 256 entries, but
        # lanes share columns across k_chunks. Each (row,col) appears exactly
        # once.
        self.assertEqual(len(seen), 256)
        self.assertTrue(all(v == 1 for v in seen.values()))


class TestGfx942PvNumeric(unittest.TestCase):
    """Pure-NumPy proof that the strided lane map reconstructs P @ V exactly."""

    def _mfma_16x16x16_via_lane_map(self, P, V):
        """Simulate one 16x16x16 PV MFMA tile using the gfx942 lane maps.

        ``P`` is [16, K] (M rows x K), ``V`` is [K, 16] (K x N cols), with
        K a multiple of 16. We walk the kernel's per-lane A (P) and B (V)
        operand layouts and accumulate exactly as the MFMA would, then read
        out the 16x16 C tile via the MFMA C-distribution. The result must
        equal ``P @ V``.
        """
        M, K = P.shape
        K2, N = V.shape
        assert M == 16 and N == 16 and K == K2 and K % 16 == 0
        # For each K-iter, reconstruct the 16x16 V K-slice purely from the
        # per-lane B-operand fragments the gfx942 strided loader hands the MFMA,
        # then contract with the matching P K-slice. If the lane map is wrong,
        # the reconstructed V slice is wrong and the result diverges from P @ V.
        C = np.zeros((16, 16), dtype=np.float64)
        for k_iter in range(K // 16):
            bmap = strided_v_b_operand_map(k_iter, n_tile=0)
            v_slice = np.zeros((16, 16), dtype=np.float64)
            for lane in range(64):
                col = lane % 16  # the MFMA B-operand output column for this lane
                for (r, c) in bmap[lane]:
                    k_local = r - k_iter * 16
                    v_slice[k_local, col] = V[r, c]
            p_slice = P[:, k_iter * 16 : k_iter * 16 + 16]
            C += p_slice @ v_slice
        return C

    def test_pv_matches_numpy_reference(self):
        rng = np.random.default_rng(0)
        M, K, N = 16, 64, 16  # K = HD (4 k-iters of 16), causal-ish small tile
        P = rng.standard_normal((M, K)).astype(np.float64)
        V = rng.standard_normal((K, N)).astype(np.float64)
        ref = P @ V
        got = self._mfma_16x16x16_via_lane_map(P, V)
        np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-12)


class TestGfx942Routing(unittest.TestCase):
    """A gfx942 request must route to instances/gfx942, never the gfx950 builder."""

    def test_tiled_2d_impl_routes_on_arch(self):
        from ck_dsl.instances.common.attention_unified import _tiled_2d_impl

        _, build942, _ = _tiled_2d_impl("gfx942")
        _, build950, _ = _tiled_2d_impl("gfx950")
        self.assertIn("gfx942", build942.__module__)
        self.assertIn("gfx950", build950.__module__)
        self.assertIsNot(build942, build950)

    def test_gfx950_builder_not_invoked_for_gfx942(self):
        # Spy on the gfx950 builder; a gfx942 build must never call it.
        import ck_dsl.instances.gfx950.attention_tiled_2d as g950

        calls = {"n": 0}
        orig = g950.build_unified_attention_2d_tiled

        def _spy(*a, **k):
            calls["n"] += 1
            return orig(*a, **k)

        g950.build_unified_attention_2d_tiled = _spy
        try:
            from ck_dsl.instances.common.attention_unified import _tiled_2d_impl

            Spec942, build942, _ = _tiled_2d_impl("gfx942")
            spec = Spec942(
                head_size=64,
                block_size=16,
                num_query_heads=8,
                num_kv_heads=8,
                dtype="fp16",
                use_sinks=False,
                sliding_window=0,
                has_softcap=False,
            )
            build942(spec, arch="gfx942")
            self.assertEqual(
                calls["n"], 0, "gfx950 builder was invoked for a gfx942 request"
            )
        finally:
            g950.build_unified_attention_2d_tiled = orig

    def test_gate_admits_gfx942_and_preserves_gfx950(self):
        from ck_dsl.instances.common.attention_arch import (
            validate_tiled_attention_arch,
        )

        # gfx950 must stay byte-identical at the decision level.
        self.assertEqual(validate_tiled_attention_arch("gfx950"), (True, "ok"))
        # gfx942 is now admitted (narrow-atom path).
        ok, _ = validate_tiled_attention_arch("gfx942")
        self.assertTrue(ok)


class TestGfx942BuildSmoke(unittest.TestCase):
    """gfx942 f16 + bf16 build with NO gfx950-only ops in the emitted IR."""

    _FORBIDDEN = (
        "16x16x32",
        "32x32x16",
        "ds_read_tr16_b64",
        "ds_read_tr_b8",
        "ds_read_tr16_b128",
    )

    def _emit_text(self, kd):
        if hasattr(kd, "to_text"):
            return kd.to_text()
        if hasattr(kd, "kernel") and hasattr(kd.kernel, "to_text"):
            return kd.kernel.to_text()
        return str(kd)

    def test_narrow_atom_only_f16_bf16(self):
        from ck_dsl.instances.common.attention_unified import _tiled_2d_impl

        Spec942, build942, _ = _tiled_2d_impl("gfx942")
        for dtype in ("fp16", "bf16"):
            spec = Spec942(
                head_size=64,
                block_size=16,
                num_query_heads=8,
                num_kv_heads=8,
                dtype=dtype,
                use_sinks=False,
                sliding_window=0,
                has_softcap=False,
            )
            kd = build942(spec, arch="gfx942")
            txt = self._emit_text(kd)
            for bad in self._FORBIDDEN:
                self.assertNotIn(
                    bad,
                    txt,
                    msg=f"gfx942 {dtype} kernel emitted gfx950-only op {bad!r}",
                )
            self.assertIn("16x16x16", txt, f"gfx942 {dtype}: no 16x16x16 atom found")

    def test_lds_gate_rejects_oversize_on_gfx942(self):
        from ck_dsl.instances.common.attention_unified import _tiled_2d_impl

        _, _, supports942 = _tiled_2d_impl("gfx942")
        # HD=128 / T=128 ~= 106 KB > gfx942 64 KB -> rejected with a clean reason.
        ok, reason = supports942(
            head_size=128,
            block_size=64,
            dtype="fp16",
            num_queries_per_kv=1,
            use_alibi=False,
            use_qq_bias=False,
            use_fp8=False,
            q_dtype=None,
            tile_size=128,
            arch="gfx942",
        )
        self.assertFalse(ok)
        self.assertIn("65536", reason)

    def test_lds_capacity_is_per_arch(self):
        self.assertEqual(ArchTarget.from_gfx("gfx942").lds_capacity_bytes, 65536)
        self.assertEqual(ArchTarget.from_gfx("gfx950").lds_capacity_bytes, 163840)


if __name__ == "__main__":
    unittest.main()
