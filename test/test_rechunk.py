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

"""Tests for :mod:`katcbf_vlbi_resample.rechunk."""

from collections.abc import Sequence
from fractions import Fraction

import numpy as np
import pytest
import xarray as xr
from astropy.time import Time

from katcbf_vlbi_resample.rechunk import Rechunk
from katcbf_vlbi_resample.utils import concat_time, isel_time

from . import SimpleStream


class TestRechunk:
    """Test :class:`.Rechunk`."""

    @pytest.fixture
    def orig_data(self, xp, cuts: Sequence[int]) -> xr.DataArray:
        """Generate input data.

        See :meth:`orig` for details.
        """
        assert len(cuts) >= 2
        n = cuts[-1] - cuts[0]
        data = xr.DataArray(
            xp.arange(2 * n).reshape((2, n)) * 2,
            dims=("pol", "time"),
            coords={"pol": ["h", "v"]},
            attrs={"time_bias": cuts[0]},
        )
        return data

    @pytest.fixture
    def orig(
        self, time_base: Time, time_scale: Fraction, orig_data: xr.DataArray, cuts: Sequence[int]
    ) -> SimpleStream[xr.DataArray]:
        """Generate input data.

        The combined data has chunk boundaries at the sample indices
        given by `cuts`. If `cuts` has n elements, there will be n - 1
        chunks.
        """
        sizes = [cuts[i] - cuts[i - 1] for i in range(1, len(cuts))]
        return SimpleStream.factory(time_base, time_scale, orig_data, sizes)

    @pytest.mark.parametrize(
        "cuts, out_cuts, partial, remainder",
        [
            (
                [50, 125, 130, 135, 550, 600, 800, 875, 950],
                [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 950],
                True,
                0,
            ),
            (
                [50, 125, 130, 135, 550, 600, 800, 875, 950],
                [100, 200, 300, 400, 500, 600, 700, 800, 900],
                False,
                0,
            ),
            (
                [100, 200],
                [100, 200],
                False,
                0,
            ),
            (
                [100, 400],
                [100, 130, 230, 330, 400],
                True,
                30,
            ),
        ],
    )
    async def test(
        self,
        orig: SimpleStream[xr.DataArray],
        orig_data: xr.DataArray,
        cuts: Sequence[int],
        out_cuts: Sequence[int],
        partial: bool,
        remainder: int,
    ) -> None:
        """Test normal usage."""
        stream = Rechunk(orig, 100, remainder=remainder, partial=partial)
        assert stream.time_base == orig.time_base
        assert stream.time_scale == orig.time_scale
        assert stream.channels is None
        assert stream.is_cupy == orig.is_cupy

        chunks = [chunk async for chunk in stream]
        data = concat_time(chunks)
        expected = orig_data
        if not partial:
            expected = isel_time(expected, np.s_[out_cuts[0] - cuts[0] : out_cuts[-1] - cuts[0]])
        xr.testing.assert_identical(data, expected)

        starts = [chunk.attrs["time_bias"] for chunk in chunks]
        assert starts == out_cuts[:-1]
