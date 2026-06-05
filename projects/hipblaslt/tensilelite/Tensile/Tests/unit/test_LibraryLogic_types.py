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
"""LibraryLogic generateLogic() pre-check tests (Step 6).

The pre-check pass lives at the top of generateLogic() before the
benchmark-data scan, so we can exercise it by invoking generateLogic
with an invalid config and a non-existent benchmarkDataPath -- the
strict gate fires (or doesn't) before the path is consulted.
"""

import os

import pytest

from Tensile.LibraryLogic import generateLogic
from Tensile.Common.TypeValidationErrors import ConfigTypeError


@pytest.fixture(autouse=True)
def _restore_env_strict_mode():
    saved = os.environ.pop("TENSILE_STRICT_TYPE_CHECK", None)
    yield
    os.environ.pop("TENSILE_STRICT_TYPE_CHECK", None)
    if saved is not None:
        os.environ["TENSILE_STRICT_TYPE_CHECK"] = saved


def _call(config, tmp_path):
    """Invoke generateLogic with a clean output dir and an unreadable input dir.

    The benchmark-data scan would happen after the strict gate; pass a
    non-existent path so we don't accidentally exercise downstream logic.
    """
    return generateLogic(
        config=config,
        benchmarkDataPath=str(tmp_path / "does_not_exist"),
        libraryLogicPath=str(tmp_path / "out"),
        cxxCompiler="hipcc",
        splitGSU=False,
        printSolutionRejectionReason=False,
        printIndexAssignmentInfo=False,
        isaInfoMap={},
    )


class TestUnknownKey:
    def test_unknown_key_raises(self, tmp_path):
        with pytest.raises(ConfigTypeError) as exc:
            _call({"DefinitelyNotAKey": 1}, tmp_path)
        assert "DefinitelyNotAKey" in str(exc.value)


class TestSolutionImportanceMin:
    def test_out_of_range_high_raises(self, tmp_path):
        with pytest.raises(ConfigTypeError) as exc:
            _call({"SolutionImportanceMin": 1.5}, tmp_path)
        assert "SolutionImportanceMin" in str(exc.value)
        assert "[0.0, 1.0]" in str(exc.value)

    def test_out_of_range_low_raises(self, tmp_path):
        with pytest.raises(ConfigTypeError) as exc:
            _call({"SolutionImportanceMin": -0.1}, tmp_path)
        assert "SolutionImportanceMin" in str(exc.value)

    def test_string_value_raises(self, tmp_path):
        with pytest.raises(ConfigTypeError) as exc:
            _call({"SolutionImportanceMin": "0.5"}, tmp_path)
        assert "SolutionImportanceMin" in str(exc.value)
        assert "expected float" in str(exc.value)

    def test_int_value_raises(self, tmp_path):
        """int is not accepted where float is expected (§5 numeric strictness rule)."""
        with pytest.raises(ConfigTypeError) as exc:
            _call({"SolutionImportanceMin": 0}, tmp_path)
        assert "SolutionImportanceMin" in str(exc.value)
        assert "expected float" in str(exc.value)


class TestLibraryType:
    def test_string_value_passes_gate(self, tmp_path):
        """Any string is accepted -- LibraryType is a distance-metric label."""
        # The gate accepts it; the scan that follows raises printExit
        # because the path doesn't exist. Use the printExit signal to
        # confirm the gate let us through. printExit raises SystemExit
        # (or RuntimeError depending on the wrapper); either way it is
        # NOT a ConfigTypeError.
        with pytest.raises(BaseException) as exc:
            _call({"LibraryType": "Equality"}, tmp_path)
        assert not isinstance(exc.value, ConfigTypeError)

    def test_int_value_raises(self, tmp_path):
        with pytest.raises(ConfigTypeError) as exc:
            _call({"LibraryType": 42}, tmp_path)
        assert "LibraryType" in str(exc.value)
        assert "expected str" in str(exc.value)


class TestCleanConfig:
    def test_clean_config_passes_gate(self, tmp_path):
        """A well-typed config passes the strict gate."""
        # Same printExit pattern as above to confirm gate let us through.
        with pytest.raises(BaseException) as exc:
            _call({
                "ScheduleName": "Test",
                "DeviceNames": "fallback",
                "ArchitectureName": "gfx942",
                "LibraryType": "GridBased",
                "SolutionImportanceMin": 0.01,
            }, tmp_path)
        assert not isinstance(exc.value, ConfigTypeError)


class TestStrictModeEnv:
    def test_off_skips_check(self, tmp_path):
        os.environ["TENSILE_STRICT_TYPE_CHECK"] = "off"
        # Bad key + bad type would raise in strict mode.
        with pytest.raises(BaseException) as exc:
            _call({"SolutionImportanceMin": 0}, tmp_path)
        assert not isinstance(exc.value, ConfigTypeError)

    def test_warn_does_not_raise(self, tmp_path, capsys):
        os.environ["TENSILE_STRICT_TYPE_CHECK"] = "warn"
        with pytest.raises(BaseException) as exc:
            _call({"SolutionImportanceMin": 0}, tmp_path)
        # Not a ConfigTypeError -- the strict gate warned and continued.
        assert not isinstance(exc.value, ConfigTypeError)
        captured = capsys.readouterr()
        assert "SolutionImportanceMin" in captured.out + captured.err


class TestMultipleErrorsAggregated:
    def test_multiple_errors_in_single_raise(self, tmp_path):
        with pytest.raises(ConfigTypeError) as exc:
            _call({
                "UnknownKey": 1,
                "SolutionImportanceMin": "bad",
                "LibraryType": 42,
            }, tmp_path)
        msg = str(exc.value)
        assert "UnknownKey" in msg
        assert "SolutionImportanceMin" in msg
        assert "LibraryType" in msg
