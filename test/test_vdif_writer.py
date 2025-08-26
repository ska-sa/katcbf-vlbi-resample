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

"""Tests for :mod:`katcbf_vlbi_resample.vdif_writer."""

import io
from fractions import Fraction

import astropy.units as u
import baseband.vdif
import cupy as cp
import numpy as np
import pytest
import xarray as xr
from astropy.time import Time, TimeDelta
from baseband.base.encoding import OPTIMAL_2BIT_HIGH
from baseband.vdif.payload import encode_2bit

from katcbf_vlbi_resample import vdif_writer
from katcbf_vlbi_resample.utils import concat_time, fraction_to_time_delta

from . import SimpleStream


@pytest.fixture
def input_data(xp) -> xr.DataArray:
    """Input data, as a single chunk."""
    return xr.DataArray(
        xp.tile(xp.arange(-2.5, 3, dtype=xp.float32), 100),
        dims=("time",),
        attrs={"time_bias": 297},
    )


@pytest.fixture
def orig(input_data: xr.DataArray, time_base: Time, time_scale: Fraction) -> SimpleStream[xr.DataArray]:
    """Input stream."""
    return SimpleStream.factory(time_base, time_scale, input_data, 10)


class TestEncode2Bit:
    """Tests for :func:`katcbf_vlbi_resample.vdif_writer._encode_2bit`."""

    def test_random(self, xp) -> None:
        """Test with random data."""
        rng = xp.random.default_rng(seed=2)
        data = rng.uniform(-5.0, 5.0, size=(100000,)).astype(xp.float32)
        expected = encode_2bit(cp.asnumpy(data))
        actual = vdif_writer._encode_2bit(data)
        # Note: values right on the threshold might round differently,
        # but do not currently seem to do so. Changes to cupy's random
        # generator may require adjustments to the test to allow for
        # a little slack.
        xp.testing.assert_array_equal(actual, xp.asarray(expected))


class TestVDIFEncode2Bit:
    """Tests for :class:`katcbf_vlbi_resample.vdif_writer.VDIFEncode2Bit."""

    def test_bad_samples_per_frame_word_align(self, orig: SimpleStream[xr.DataArray]) -> None:
        """Test that `samples_per_frame` not a multiple of word size raises :exc:`ValueError`."""
        with pytest.raises(ValueError, match="samples_per_frame must be a multiple of 32"):
            vdif_writer.VDIFEncode2Bit(orig, 160016)

    def test_bad_samples_per_frame_rate(self, orig: SimpleStream[xr.DataArray]) -> None:
        """Test that ValueError is raised if frame rate is not an integer."""
        with pytest.raises(ValueError, match="samples_per_frame does not yield an integer frame rate"):
            vdif_writer.VDIFEncode2Bit(orig, 160032)

    async def test_success(self, xp, orig: SimpleStream[xr.DataArray], input_data: xr.DataArray) -> None:
        """Test normal usage."""
        enc = vdif_writer.VDIFEncode2Bit(orig, 160)
        assert enc.time_scale == orig.time_scale * vdif_writer.VDIFEncode2Bit.SAMPLES_PER_WORD
        assert enc.is_cupy == orig.is_cupy
        chunks = [chunk async for chunk in enc]
        out_data = concat_time(chunks)

        # The original time_base is on a frame boundary, so frame boundaries
        # occur when the sample index is a multiple of samples_per_frame. So
        # after discarding a partial frame, the output should start at sample
        # index 320 on the original clock.
        expected_start = orig.time_base + fraction_to_time_delta(320 * orig.time_scale)
        actual_start = enc.time_base + fraction_to_time_delta(out_data.attrs["time_bias"] * enc.time_scale)
        assert abs(actual_start - expected_start) <= TimeDelta(1e-10, format="sec")
        assert out_data.sizes["time"] == 480 // vdif_writer.VDIFEncode2Bit.SAMPLES_PER_WORD
        used_input_data = input_data.isel(time=xp.s_[23:503])
        xp.testing.assert_array_equal(out_data.data, vdif_writer._encode_2bit_words(used_input_data.data))


class TestVDIFFormatter:
    """Tests for :class:`katcbf_vlbi_resample.vdif_writer.VDIFFormatter`."""

    def test_channelised(self, time_base: Time, time_scale: Fraction) -> None:
        """Test that passing a channelised input raises an exception."""
        stream = SimpleStream.factory(
            time_base,
            time_scale,
            xr.DataArray(np.zeros((16, 16)), dims=("channel", "time"), attrs={"time_bias": 0}),
        )
        with pytest.raises(ValueError, match="unchannelised"):
            vdif_writer.VDIFFormatter(stream, [{}], station="me", samples_per_frame=80)

    def test_bad_samples_per_frame_word_align(self, orig: SimpleStream[xr.DataArray]) -> None:
        """Test that `samples_per_frame` not a multiple of word size raises :exc:`ValueError`."""
        with pytest.raises(ValueError, match="samples_per_frame must be a multiple of 32"):
            vdif_writer.VDIFFormatter(orig, [{}], station="me", samples_per_frame=160016)

    def test_bad_samples_per_frame_rate(self, orig: SimpleStream[xr.DataArray]) -> None:
        """Test that ValueError is raised if frame rate is not an integer."""
        with pytest.raises(ValueError, match="samples_per_frame does not yield an integer frame rate"):
            vdif_writer.VDIFFormatter(orig, [{}], station="me", samples_per_frame=160032)

    async def test_success(self, xp, time_base: Time, time_scale: Fraction) -> None:
        """Test normal usage."""
        rng = np.random.default_rng(seed=1)
        # We test separately that VDIFEncoder2Bit clips data to whole frames,
        # so for this test we keep things simple by aligning everything to
        # frames. We also stick to values that round-trip through quantisation
        # (the float32 type is needed for that to avoid rounding differences).
        # cupy's RNG doesn't support `choice`, so we generate the values on the
        # CPU then transfer them to cupy if necessary.
        data = xr.DataArray(
            xp.asarray(
                rng.choice(
                    np.array([-OPTIMAL_2BIT_HIGH, -1.0, 1.0, OPTIMAL_2BIT_HIGH], np.float32),
                    size=(2, 160000),
                ),
            ),
            dims=("pol", "time"),
            coords={"pol": ["v", "h"]},
            attrs={"time_bias": 320},
        )
        samples_per_frame = 160
        orig = SimpleStream.factory(time_base, time_scale, data, 100)
        enc = vdif_writer.VDIFEncode2Bit(orig, samples_per_frame)
        fmt = vdif_writer.VDIFFormatter(
            enc, [{"pol": "h"}, {"pol": "v"}], station="me", samples_per_frame=samples_per_frame
        )

        # Write the data to an in-memory file
        fh = io.BytesIO()
        async for frameset in fmt:
            frameset.tofile(fh)

        # Read it back and compare to the original data
        fh.seek(0, 0)
        with baseband.vdif.open(fh, "rs") as vdif_data:
            assert vdif_data.sample_rate == float(1 / time_scale) * u.Hz
            assert vdif_data.samples_per_frame == samples_per_frame
            assert vdif_data.bps == 2
            assert vdif_data.shape == data.shape[::-1]
            assert vdif_data.sample_shape == (2,)
            assert (vdif_data.start_time - time_base).sec == pytest.approx(0.002, abs=1e-10)
            assert vdif_data.header0.station == "me"
            out_data = xr.DataArray(
                vdif_data.read(),
                dims=("time", "pol"),
                coords={"pol": ["h", "v"]},  # Order of `threads`
            )
            # Align out_data to match the axes of data
            out_data = out_data.reindex_like(data).transpose(*data.dims)
            xr.testing.assert_equal(out_data, data.as_numpy())
