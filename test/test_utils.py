################################################################################
# Copyright (c) 2024, National Research Foundation (SARAO)
#
# Licensed under the BSD 3-Clause License (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy
# of the License at
#
#   https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

"""Tests for :mod:`katcbf_vlbi_resample.utils."""

from fractions import Fraction

import pytest
import xarray as xr
from astropy.time import TimeDelta

from katcbf_vlbi_resample.utils import concat_time, fraction_to_time_delta, isel_time, time_align


@pytest.fixture
def array(xp) -> xr.DataArray:
    """Build an array to use as a fixture."""
    return xr.DataArray(
        xp.array([3, 1, 4, 1, 5, 9, 2, 6, 5]),
        dims=("time",),
        attrs={"time_bias": 100},
    )


class TestConcatTime:
    """Tests for :func:`katcbf_vlbi_resample.utils.concat_time`."""

    @pytest.fixture
    def array1(self, xp) -> xr.DataArray:
        """Create a 2x5 array."""
        return xr.DataArray(
            xp.arange(10, 20, dtype=xp.uint8).reshape(2, 5),
            dims=("pol", "time"),
            coords={"pol": ["h", "v"]},
            attrs={"time_bias": 33},
        )

    @pytest.fixture
    def array2(self, xp) -> xr.DataArray:
        """Create a 2x5 array, with different axis ordering to :meth:`array1`."""
        return xr.DataArray(
            xp.ones((3, 2), dtype=xp.uint8),
            dims=("time", "pol"),
            coords={"pol": ["h", "v"]},
            attrs={"time_bias": 38},
        )

    def test_success(self, xp, array1: xr.DataArray, array2: xr.DataArray) -> None:
        """Test the normal usage path."""
        out = concat_time([array1, array2])
        assert out.attrs["time_bias"] == 33
        assert out.dims == ("pol", "time")
        assert out.shape == (2, 8)
        xp.testing.assert_array_equal(
            out.data,
            [[10, 11, 12, 13, 14, 1, 1, 1], [15, 16, 17, 18, 19, 1, 1, 1]],
        )

    def test_non_contiguous(self, array1: xr.DataArray, array2: xr.DataArray) -> None:
        """Test that an exception is raised if the `time_bias` fields are inconsistent."""
        with pytest.raises(ValueError):
            concat_time([array2, array1])
        with pytest.raises(ValueError):
            concat_time([array2, array2])

    def test_mismatched_coords(self, array1: xr.DataArray, array2: xr.DataArray) -> None:
        """Test that an exception is raised if the non-time axes aren't aligned."""
        array2.coords["pol"] = ["l", "r"]
        with pytest.raises(ValueError):
            concat_time([array2, array2])


class TestIselTime:
    """Tests for :func:`katcbf_vlbi_resample.utils.isel_time`."""

    def test_basic(self, xp, array: xr.DataArray) -> None:
        """Test simplest case."""
        out = isel_time(array, xp.s_[3:6])
        xp.testing.assert_array_equal(out.data, [1, 5, 9])
        assert out.attrs["time_bias"] == 103

    def test_explicit_step(self, xp, array: xr.DataArray) -> None:
        """Test that explicit step size of 1 is accepted."""
        out = isel_time(array, xp.s_[3:6:1])
        xp.testing.assert_array_equal(out.data, [1, 5, 9])
        assert out.attrs["time_bias"] == 103

    def test_empty_start(self, xp, array: xr.DataArray) -> None:
        """Test that an unspecified start index works as expected."""
        out = isel_time(array, xp.s_[:4])
        xp.testing.assert_array_equal(out.data, [3, 1, 4, 1])
        assert out.attrs["time_bias"] == 100

    def test_empty_end(self, xp, array: xr.DataArray) -> None:
        """Test that an unspecified end index works as expected."""
        out = isel_time(array, xp.s_[4:])
        xp.testing.assert_array_equal(out.data, [5, 9, 2, 6, 5])
        assert out.attrs["time_bias"] == 104

    def test_negative_start(self, xp, array: xr.DataArray) -> None:
        """Test that a negative start index works as expected."""
        out = isel_time(array, xp.s_[-6:6])
        xp.testing.assert_array_equal(out.data, [1, 5, 9])
        assert out.attrs["time_bias"] == 103

    def test_bad_step(self, xp, array: xr.DataArray) -> None:
        """Test that a non-unity step raises an exception."""
        with pytest.raises(ValueError):
            isel_time(array, xp.s_[3:6:2])


class TestTimeAlign:
    """Tests for :func:`katcbf_vlbi_resample.utils.time_align`."""

    def test_trim(self, xp, array: xr.DataArray) -> None:
        """Test the case where some data is trimmed."""
        out = time_align(array, 10, 2)
        assert out is not None
        xp.testing.assert_array_equal(out.data, [4, 1, 5, 9, 2, 6, 5])
        assert out.attrs["time_bias"] == 102

    def test_no_trim(self, array: xr.DataArray) -> None:
        """Test that case where no trimming is needed."""
        assert time_align(array, 10) is array

    def test_none(self, array: xr.DataArray) -> None:
        """Test that case that the array doesn't contain a boundary."""
        assert time_align(array, 10, 9) is None


class TestFractionToTimeDelta:
    """Tests for :func:`katcbf_vlbi_resample.utils.fraction_to_time_delta`."""

    def test_simple(self) -> None:
        """Test the basic functionality."""
        f = 3 + Fraction(1, 4)
        td = fraction_to_time_delta(f)
        assert td.scale == "tai"
        assert td.sec == pytest.approx(3.25, rel=1e-12)

    def test_precision(self) -> None:
        """Test that there is better than nanosecond precision over large time spans."""
        f = 1_000_000 + Fraction(123_456_789, 1_000_000_000)
        td = fraction_to_time_delta(f)
        td -= TimeDelta(1_000_000, format="sec", scale="tai")
        assert td.sec == pytest.approx(0.123456789, abs=1e-10)
