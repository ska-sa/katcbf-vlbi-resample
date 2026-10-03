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

    async def test(self, time_base: Time, time_scale: Fraction) -> None:
        """Test basic functionality."""
        data = xr.DataArray(np.arange(1000), dims=("time",), attrs={"time_bias": 100})
        orig = SimpleStream.factory(time_base, time_scale, data, 5)
        stream = AsCupy(orig)
        assert stream.time_base == time_base
        assert stream.time_scale == time_scale
        assert stream.channels is None
        assert stream.is_cupy

        chunks = [chunk async for chunk in stream]
        for chunk in chunks:
            assert isinstance(chunk.data, cp.ndarray)
        out = concat_time(chunks)
        xr.testing.assert_equal(as_cupy(data), out)


class TestAsNumpy:
    """Tests for :class:`katcbf_vlbi_resample.cupy_bridge.AsNumpy`."""

    async def test(self, time_base: Time, time_scale: Fraction) -> None:
        """Test basic functionality."""
        data = xr.DataArray(cp.arange(1000), dims=("time",), attrs={"time_bias": 100})
        orig = SimpleStream.factory(time_base, time_scale, data, 5)
        stream = AsNumpy(orig)
        assert stream.time_base == time_base
        assert stream.time_scale == time_scale
        assert stream.channels is None
        assert not stream.is_cupy

        chunks = [chunk async for chunk in stream]
        for chunk in chunks:
            assert isinstance(chunk.data, np.ndarray)
        out = concat_time(chunks)
        xr.testing.assert_equal(data.as_numpy(), out)
