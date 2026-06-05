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
"""Strict type-check coverage for assignGlobalParameters (Step 5)."""

import os

import pytest

from Tensile.Common.GlobalParameters import (
    assignGlobalParameters,
    globalParameters,
    globalParameterTypeOverrides,
    restoreDefaultGlobalParameters,
    _assertOverrideTableCovers,
)
from Tensile.Common.TypeValidationErrors import ConfigTypeError


# A minimal isaInfoMap stand-in. assignGlobalParameters uses it to set
# validParameters["ISA"] and (optionally) print capability tables; a
# dict with no keys is the smallest sufficient input for the validation
# paths exercised here.
EMPTY_ISA_INFO = {}


@pytest.fixture(autouse=True)
def _restore_env_strict_mode():
    saved = os.environ.pop("TENSILE_STRICT_TYPE_CHECK", None)
    yield
    os.environ.pop("TENSILE_STRICT_TYPE_CHECK", None)
    if saved is not None:
        os.environ["TENSILE_STRICT_TYPE_CHECK"] = saved


@pytest.fixture(autouse=True)
def _restore_global_defaults():
    """Reset globalParameters after each test so order-dependence vanishes."""
    yield
    restoreDefaultGlobalParameters()


class TestBoolIntTrap:
    def test_bool_where_int_expected_raises(self):
        """BoundsCheck default is int 0 -- a bool must be rejected."""
        with pytest.raises(ConfigTypeError) as exc:
            assignGlobalParameters({"BoundsCheck": False}, EMPTY_ISA_INFO)
        assert "BoundsCheck" in str(exc.value)
        assert "bool" in str(exc.value)
        assert "expected int" in str(exc.value)

    def test_int_where_bool_expected_raises(self):
        """PinClocks default is bool False -- an int must be rejected."""
        with pytest.raises(ConfigTypeError) as exc:
            assignGlobalParameters({"PinClocks": 1}, EMPTY_ISA_INFO)
        assert "PinClocks" in str(exc.value)
        assert "expected bool" in str(exc.value)

    def test_valid_int_passes(self):
        assignGlobalParameters({"BoundsCheck": 1}, EMPTY_ISA_INFO)
        assert globalParameters["BoundsCheck"] == 1

    def test_valid_bool_passes(self):
        assignGlobalParameters({"PinClocks": True}, EMPTY_ISA_INFO)
        assert globalParameters["PinClocks"] is True


class TestUnknownKeyRaises:
    def test_unknown_global_raises(self):
        """Step 5 promotes the unknown-key printWarning to a raise."""
        with pytest.raises(ConfigTypeError) as exc:
            assignGlobalParameters({"DefinitelyNotAGlobal": 42}, EMPTY_ISA_INFO)
        assert "DefinitelyNotAGlobal" in str(exc.value)
        assert "Unknown global parameter" in str(exc.value)

    def test_ignore_keys_pass_through(self):
        """Keys listed in ignoreKeys must not raise (they aren't globals)."""
        # 'OutputPath' is in the ignoreKeys list (used by RetuneLibrary).
        # Pass a path-typed value: passing through cleanly is the test.
        from pathlib import Path
        assignGlobalParameters({"OutputPath": Path("/tmp/x")}, EMPTY_ISA_INFO)


class TestOverrideTable:
    def test_int_for_none_default_raises(self):
        """RocProfCounter default is None -> {NoneType, str}. Int is rejected."""
        with pytest.raises(ConfigTypeError) as exc:
            assignGlobalParameters({"RocProfCounter": 42}, EMPTY_ISA_INFO)
        assert "RocProfCounter" in str(exc.value)
        assert "expected NoneType or str" in str(exc.value)

    def test_none_for_none_default_passes(self):
        assignGlobalParameters({"RocProfCounter": None}, EMPTY_ISA_INFO)
        assert globalParameters["RocProfCounter"] is None

    def test_string_for_none_default_passes(self):
        assignGlobalParameters({"RocProfCounter": "WAVES"}, EMPTY_ISA_INFO)
        assert globalParameters["RocProfCounter"] == "WAVES"

    def test_override_table_has_all_none_defaults(self):
        """Every None-defaulted globalParameters key has an entry here."""
        none_defaults = {k for k, v in globalParameters.items() if v is None}
        missing = none_defaults - set(globalParameterTypeOverrides.keys())
        assert not missing, f"Missing override entries: {missing}"

    def test_coverage_assertion_catches_missing_entry(self):
        """_assertOverrideTableCovers raises if a None default lacks an override."""
        defaults = {"X": None, "Y": "string"}
        with pytest.raises(RuntimeError) as exc:
            _assertOverrideTableCovers(defaults, {})
        assert "'X'" in str(exc.value)

    def test_coverage_assertion_passes_with_all_annotated(self):
        defaults = {"X": None, "Y": "string"}
        _assertOverrideTableCovers(defaults, {"X": {type(None), str}})


class TestMinimumRequiredVersionSkipped:
    def test_minimum_required_version_passes(self):
        # MinimumRequiredVersion is handled separately at the top of
        # assignGlobalParameters; the type loop must skip it. (Default
        # is a string, this would otherwise be a string-vs-string no-op,
        # but a compatible version string should also pass without
        # surprises.)
        from Tensile import __version__
        assignGlobalParameters({"MinimumRequiredVersion": __version__}, EMPTY_ISA_INFO)


class TestMultipleMismatchesAggregated:
    def test_multiple_errors_in_single_raise(self):
        with pytest.raises(ConfigTypeError) as exc:
            assignGlobalParameters(
                {
                    "BoundsCheck": False,    # bad: bool for int
                    "PinClocks": 1,          # bad: int for bool
                    "RocProfCounter": 42,    # bad: int for None|str
                },
                EMPTY_ISA_INFO,
            )
        msg = str(exc.value)
        assert "BoundsCheck" in msg
        assert "PinClocks" in msg
        assert "RocProfCounter" in msg


class TestStrictModeEnv:
    def test_off_skips_type_check(self):
        os.environ["TENSILE_STRICT_TYPE_CHECK"] = "off"
        # Would raise in strict mode.
        assignGlobalParameters({"BoundsCheck": False}, EMPTY_ISA_INFO)
        assert globalParameters["BoundsCheck"] is False

    def test_warn_does_not_raise(self, capsys):
        os.environ["TENSILE_STRICT_TYPE_CHECK"] = "warn"
        assignGlobalParameters({"BoundsCheck": False}, EMPTY_ISA_INFO)
        captured = capsys.readouterr()
        assert "BoundsCheck" in captured.out + captured.err

    def test_strict_explicit_raises(self):
        os.environ["TENSILE_STRICT_TYPE_CHECK"] = "strict"
        with pytest.raises(ConfigTypeError):
            assignGlobalParameters({"BoundsCheck": False}, EMPTY_ISA_INFO)
