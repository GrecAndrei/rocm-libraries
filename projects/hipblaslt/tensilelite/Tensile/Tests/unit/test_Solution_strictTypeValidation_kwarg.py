################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
################################################################################
"""Step 8 (B4): strictTypeValidation kwarg threading.

Verifies that Solution.__init__'s new strictTypeValidation kwarg
controls whether validateParameterTypes runs on the post-construction
state, that the input-YAML callers in BenchmarkProblems pass False,
that the printTypeMismatchSummary import/call was removed, and that
the library-logic path (LibraryIO) leaves the default True.
"""

import inspect

import pytest

from Tensile.SolutionStructs.Solution import Solution


class TestSolutionKwarg:
    def test_init_has_kwarg(self):
        sig = inspect.signature(Solution.__init__)
        assert "strictTypeValidation" in sig.parameters

    def test_init_kwarg_defaults_true(self):
        sig = inspect.signature(Solution.__init__)
        assert sig.parameters["strictTypeValidation"].default is True


class TestBenchmarkProblemsCallSitesPassFalse:
    """The two input-YAML Solution() call sites pass strictTypeValidation=False."""

    def test_generate_single_solution_passes_false(self):
        src = inspect.getsource(_generate_single_solution())
        assert "strictTypeValidation=False" in src

    def test_get_custom_kernel_solution_obj_passes_false(self):
        src = inspect.getsource(_get_custom_kernel_solution_obj())
        assert "strictTypeValidation=False" in src


class TestPrintSummaryRemovedFromBenchmarkProblems:
    """printTypeMismatchSummary call/import is gone from BenchmarkProblems.py."""

    def test_import_removed(self):
        from Tensile import BenchmarkProblems
        # The name should no longer be a module attribute.
        assert not hasattr(BenchmarkProblems, "printTypeMismatchSummary")

    def test_call_site_removed(self):
        from Tensile import BenchmarkProblems
        src = inspect.getsource(BenchmarkProblems)
        assert "printTypeMismatchSummary()" not in src


class TestLibraryIODefaultsToTrue:
    """LibraryIO callers do NOT pass strictTypeValidation -> default True applies."""

    def test_parse_solutions_data_does_not_pass_kwarg(self):
        from Tensile import LibraryIO
        src = inspect.getsource(LibraryIO)
        # Make sure neither LibraryIO call site opts out.
        # The default behaviour (True) is what we want for the
        # library-logic path so the collector continues to populate.
        assert "strictTypeValidation=False" not in src


def _generate_single_solution():
    from Tensile import BenchmarkProblems
    return BenchmarkProblems._generate_single_solution


def _get_custom_kernel_solution_obj():
    from Tensile import BenchmarkProblems
    return BenchmarkProblems._getCustomKernelSolutionObj
