# Copyright (c) 2024, National Research Foundation (SARAO)

"""Tests for :mod:`katcbf_vlbi_resample.utils."""

from fractions import Fraction

import pytest
import xarray as xr

from katcbf_vlbi_resample.utils import concat_time


class TestConcatTime:
    """Tests for :func:`katcbf_vlbi_resample.utils.concat_time`."""

    @pytest.fixture
    def array1(self, xp) -> xr.DataArray:
        """Create a 2x5 array."""
        return xr.DataArray(
            xp.arange(10, 20, dtype=xp.uint8).reshape(2, 5),
            dims=("pol", "time"),
            coords={"pol": ["h", "v"]},
            attrs={"time_bias": Fraction(100, 3)},
        )

    @pytest.fixture
    def array2(self, xp) -> xr.DataArray:
        """Create a 2x5 array, with different axis ordering to :meth:`array1`."""
        return xr.DataArray(
            xp.ones((3, 2), dtype=xp.uint8),
            dims=("time", "pol"),
            coords={"pol": ["h", "v"]},
            attrs={"time_bias": Fraction(115, 3)},
        )

    def test_success(self, xp, array1: xr.DataArray, array2: xr.DataArray) -> None:
        """Test the normal usage path."""
        out = concat_time([array1, array2])
        assert out.attrs["time_bias"] == Fraction(100, 3)
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
