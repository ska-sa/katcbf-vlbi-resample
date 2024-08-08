# Copyright (c) 2024, National Research Foundation (SARAO)

"""Tests for :mod:`katcbf_vlbi_resample.cupy_bridge."""

from fractions import Fraction

import cupy as cp
import numpy as np
import xarray as xr
from astropy.time import Time

from katcbf_vlbi_resample.cupy_bridge import AsCupy, AsNumpy
from katcbf_vlbi_resample.utils import as_cupy, concat_time

from . import SimpleStream


class TestAsCupy:
    """Tests for :class:`katcbf_vlbi_resample.cupy_bridge.AsCupy`."""

    def test(self, time_base: Time, time_scale: Fraction) -> None:
        """Test basic functionality."""
        data = xr.DataArray(np.arange(1000), dims=("time",), attrs={"time_bias": Fraction(100)})
        orig = SimpleStream(time_base, time_scale, data, 5)
        stream = AsCupy(orig)
        assert stream.time_base == time_base
        assert stream.time_scale == time_scale
        assert stream.channels is None
        assert stream.is_cupy

        chunks = list(stream)
        for chunk in chunks:
            assert isinstance(chunk.data, cp.ndarray)
        out = concat_time(chunks)
        xr.testing.assert_equal(as_cupy(data), out)


class TestAsNumpy:
    """Tests for :class:`katcbf_vlbi_resample.cupy_bridge.AsNumpy`."""

    def test(self, time_base: Time, time_scale: Fraction) -> None:
        """Test basic functionality."""
        data = xr.DataArray(cp.arange(1000), dims=("time",), attrs={"time_bias": Fraction(100)})
        orig = SimpleStream(time_base, time_scale, data, 5)
        stream = AsNumpy(orig)
        assert stream.time_base == time_base
        assert stream.time_scale == time_scale
        assert stream.channels is None
        assert not stream.is_cupy

        chunks = list(stream)
        for chunk in chunks:
            assert isinstance(chunk.data, np.ndarray)
        out = concat_time(chunks)
        xr.testing.assert_equal(data.as_numpy(), out)
