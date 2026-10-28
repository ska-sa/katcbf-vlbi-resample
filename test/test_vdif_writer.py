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
from baseband.base.encoding import OPTIMAL_2BIT_HIGH, TWO_BIT_1_SIGMA
from baseband.vdif.payload import encode_2bit

from katcbf_vlbi_resample import vdif_writer
from katcbf_vlbi_resample.utils import fraction_to_time_delta

from . import SimpleStream

N_POLS = 2
N_STREAMS = 4


# astropy warns about times in the distant future (presumably because
# leap-seconds are not known).
@pytest.mark.filterwarnings("ignore:.*dubious year.*")
@pytest.mark.parametrize(
    "time, ref_time, epoch",
    [
        ("2000-01-01 00:00:00", "2000-01-01 00:00:00", 0),
        ("2015-06-30 23:59:60", "2015-01-01 00:00:00", 30),  # Leap second
        ("2025-03-02 12:34:56", "2025-01-01 00:00:00", 50),
        ("2025-09-01 12:34:56", "2025-07-01 00:00:00", 51),
        ("2032-01-01 00:00:00", "2032-01-01 00:00:00", 64),  # _reference_epoch shouldn't wrap at 6 bits
    ],
)
def test_reference_epoch(time: str, ref_time: str, epoch: int) -> None:
    """Test :func:`katcbf_vlbi_resample.vdif_writer._reference_epoch."""
    time_a = Time(time, format="iso", scale="utc")
    ref_time_a = Time(ref_time, format="iso", scale="utc")
    assert vdif_writer._reference_epoch(time_a) == (ref_time_a, epoch)


def test_make_header() -> None:
    """Test :func:`katcbf_vlbi_resample.vdif_writer._make_header.

    There is also indirect testing in TestVDIFFormatter which covers cases not
    tested here (particularly non-numeric station IDs).
    """
    header, ref_time = vdif_writer._make_header(
        bps=2,
        station=1234,
        samples_per_frame=96,
        ref_time=Time("2025-09-01 12:34:56", format="iso", scale="utc"),
    )
    assert ref_time == Time("2025-07-01 00:00:00", format="iso", scale="utc")
    parsed = baseband.vdif.header.VDIFHeader(header)
    assert parsed.bps == 2
    assert parsed.station == 1234
    assert parsed.samples_per_frame == 96
    assert parsed["ref_epoch"] == 51
    # This field is intentionally set to 0 (indicating valid data)
    assert parsed["invalid_data"] == 0
    # Check that fields that are meant to be blank in fact are
    assert parsed["thread_id"] == 0
    assert parsed["seconds"] == 0
    assert parsed["frame_nr"] == 0


def test_make_header_bad() -> None:
    """Test exception handling in :func:`katcbf_vlbi_resample.vdif_writer._make_header`."""
    ref_time = Time("2025-09-01 12:34:56", format="iso", scale="utc")
    with pytest.raises(ValueError, match="bits/sample does not fit in the 5-bit VDIF header field"):
        vdif_writer._make_header(bps=33, station="Me", samples_per_frame=256, ref_time=ref_time)
    with pytest.raises(ValueError, match="numeric station ID is out of range"):
        vdif_writer._make_header(bps=2, station=-1, samples_per_frame=96, ref_time=ref_time)
    with pytest.raises(ValueError, match="numeric station ID is out of range"):
        vdif_writer._make_header(bps=2, station=0x3000, samples_per_frame=96, ref_time=ref_time)
    with pytest.raises(ValueError, match="samples_per_frame does not yield a whole number of 8-byte units"):
        vdif_writer._make_header(bps=3, station="Me", samples_per_frame=96, ref_time=ref_time)


@pytest.fixture
def input_data(xp) -> xr.DataArray:
    """Input data, as a single chunk."""
    return xr.DataArray(
        xp.tile(xp.arange(-2.5, 3, dtype=xp.float32), 100),
        dims=("time",),
        attrs={"time_bias": 297},
    )


@pytest.fixture
def orig_samples(input_data: xr.DataArray, time_base: Time, time_scale: Fraction) -> SimpleStream[xr.DataArray]:
    """Input stream."""
    return SimpleStream.factory(time_base, time_scale, input_data, 10)


@pytest.fixture
def samples_per_frame() -> int:
    """Return the default number of samples per VDIF frame."""
    return 160


@pytest.fixture
def frame_time_scale(samples_per_frame: int, time_scale: Fraction) -> Fraction:
    """Time scale that yields a non-integer frame rate for a given `samples_per_frame`."""
    return time_scale * samples_per_frame


@pytest.fixture
def aligned_time_base(time_base: Time) -> Time:
    """Time base shifted back to the start of its UTC second.

    This is specifically used in TestVDIFFormatter, where it bypasses VDIFEncode2Bit.
    """
    fractional_second = time_base.utc.ymdhms.second % 1
    return time_base - TimeDelta(fractional_second, format="sec", scale="tai")


@pytest.fixture
def input_dataset(xp, input_data: xr.DataArray) -> xr.Dataset:
    """Input dataset, whose data-vars have a single (`time`) dimension."""
    return xr.Dataset(
        {
            "data": input_data,
            "invalid_data": xr.DataArray(
                xp.zeros(input_data.sizes["time"], dtype=xp.bool_),
                dims=input_data.dims,
            ),
        },
        attrs=input_data.attrs,  # This is done to navigate utils.isel_time
    )


@pytest.fixture
def orig_frames(
    input_dataset: xr.Dataset, aligned_time_base: Time, frame_time_scale: Fraction
) -> SimpleStream[xr.Dataset]:
    """Input stream as `xr.Dataset`.

    This is specifically used in TestVDIFFormatter, where it bypasses VDIFEncode2Bit.
    """
    return SimpleStream.factory(aligned_time_base, frame_time_scale, input_dataset, 10)


class TestEncode2Bit:
    """Tests for :func:`katcbf_vlbi_resample.vdif_writer._encode_2bit`."""

    def test_random(self, xp) -> None:
        """Test with random data."""
        rng = xp.random.default_rng(seed=2)
        data = rng.uniform(-5.0, 5.0, size=(100000,)).astype(xp.float32)
        expected = encode_2bit(cp.asnumpy(data))
        actual = vdif_writer._encode_2bit(data, TWO_BIT_1_SIGMA)
        # Note: values right on the threshold might round differently,
        # but do not currently seem to do so. Changes to cupy's random
        # generator may require adjustments to the test to allow for
        # a little slack.
        xp.testing.assert_array_equal(actual, xp.asarray(expected))


class TestVDIFEncode2Bit:
    """Tests for :class:`katcbf_vlbi_resample.vdif_writer.VDIFEncode2Bit."""

    @pytest.mark.parametrize("samples_per_frame", [160016])
    def test_bad_samples_per_frame_word_align(
        self, samples_per_frame: int, orig_samples: SimpleStream[xr.DataArray]
    ) -> None:
        """Test that `samples_per_frame` not a multiple of word size raises :exc:`ValueError`."""
        with pytest.raises(ValueError, match="samples_per_frame must be a multiple of 32"):
            vdif_writer.VDIFEncode2Bit(orig_samples, samples_per_frame, 1.0)

    @pytest.mark.parametrize("samples_per_frame", [160032])
    def test_bad_samples_per_frame_rate(self, orig_samples: SimpleStream[xr.DataArray], samples_per_frame: int) -> None:
        """Test that ValueError is raised if frame rate is not an integer."""
        with pytest.raises(ValueError, match="samples_per_frame does not yield an integer frame rate"):
            vdif_writer.VDIFEncode2Bit(orig_samples, samples_per_frame, 1.0)

    async def test_success(
        self, xp, samples_per_frame: int, orig_samples: SimpleStream[xr.DataArray], input_data: xr.DataArray
    ) -> None:
        """Test normal usage."""
        words_per_frame = samples_per_frame // vdif_writer.VDIFEncode2Bit.SAMPLES_PER_WORD
        enc = vdif_writer.VDIFEncode2Bit(orig_samples, samples_per_frame, 1.0)
        # NOTE: Encoded data now has shape (..., time, words_per_frame)
        assert enc.time_scale == orig_samples.time_scale * samples_per_frame
        assert enc.is_cupy == orig_samples.is_cupy
        chunks = [chunk async for chunk in enc]
        # Concatenate encoded data across the frame dimension
        concat_data = xr.concat([chunk["data"] for chunk in chunks], dim="time")

        # The original time_base is on a frame boundary, so frame boundaries
        # occur when the sample index is a multiple of samples_per_frame. So
        # after discarding a partial frame, the output should start at sample
        # index 320 on the original clock.
        expected_start = orig_samples.time_base + fraction_to_time_delta(320 * orig_samples.time_scale)
        actual_start = enc.time_base + fraction_to_time_delta(concat_data.attrs["time_bias"] * enc.time_scale)
        assert abs(actual_start - expected_start) <= TimeDelta(1e-10, format="sec")
        assert (
            concat_data.sizes["time"] * concat_data.sizes["word"] == 480 // vdif_writer.VDIFEncode2Bit.SAMPLES_PER_WORD
        )
        used_input_data = input_data.isel(time=xp.s_[23:503])
        expected_encoded = vdif_writer._encode_2bit_words(used_input_data.data, 1.0)
        expected_encoded = expected_encoded.reshape(-1, words_per_frame)
        xp.testing.assert_array_equal(concat_data.data, expected_encoded)

    async def test_nan_sets_per_frame_invalid_data(
        self, xp, samples_per_frame: int, time_base: Time, time_scale: Fraction
    ) -> None:
        """Test each frame that contains at least one NaN is marked as invalid."""
        n_frames = 3
        data = xp.zeros(shape=(N_STREAMS, n_frames * samples_per_frame, N_POLS), dtype=xp.float32)
        frame_midpoint_index = samples_per_frame // 2
        # NOTE: We keep NaNs localized to individual (stream, pol) combinations
        # so the test verifies that each combination's affected frame is marked
        # invalid independently. This is achieved by marking one NaN in the first
        # frame and another in the third frame.
        data[1, frame_midpoint_index, 0] = xp.nan
        data[3, 2 * samples_per_frame + frame_midpoint_index, 1] = xp.nan
        input_data = xr.DataArray(
            data,
            dims=("stream", "time", "pol"),
            coords={"stream": list(range(N_STREAMS)), "pol": ["X", "Y"]},
            attrs={"time_bias": 0},
        )
        orig_samples = SimpleStream.factory(time_base, time_scale, input_data)
        enc = vdif_writer.VDIFEncode2Bit(orig_samples, samples_per_frame, TWO_BIT_1_SIGMA)

        chunks = [chunk async for chunk in enc]
        invalid_data = xr.concat([chunk["invalid_data"] for chunk in chunks], dim="time")
        expected = xp.zeros((N_STREAMS, n_frames, N_POLS), dtype=xp.bool_)
        expected[1, 0, 0] = True
        expected[3, 2, 1] = True
        assert invalid_data.dims == ("stream", "time", "pol")
        xr.testing.assert_equal(invalid_data.coords["stream"], input_data.coords["stream"])
        xr.testing.assert_equal(invalid_data.coords["pol"], input_data.coords["pol"])
        xp.testing.assert_array_equal(invalid_data.data, expected)


class TestVDIFFormatter:
    """Tests for :class:`katcbf_vlbi_resample.vdif_writer.VDIFFormatter`."""

    def test_channelised(self, time_base: Time, time_scale: Fraction) -> None:
        """Test that passing a channelised input raises an exception."""
        stream = SimpleStream.factory(
            time_base,
            time_scale,
            xr.Dataset({"data": xr.DataArray(np.zeros((16, 16)), dims=("channel", "time"))}, attrs={"time_bias": 0}),
        )
        with pytest.raises(ValueError, match="unchannelised"):
            vdif_writer.VDIFFormatter(stream, [{}], station="Me", samples_per_frame=80)

    @pytest.mark.parametrize("samples_per_frame", [160016])
    def test_bad_samples_per_frame_word_align(
        self, samples_per_frame: int, orig_frames: SimpleStream[xr.Dataset]
    ) -> None:
        """Test that `samples_per_frame` not a multiple of word size raises :exc:`ValueError`."""
        with pytest.raises(ValueError, match="samples_per_frame must be a multiple of 32"):
            vdif_writer.VDIFFormatter(orig_frames, [{}], station="Me", samples_per_frame=samples_per_frame)

    @pytest.mark.parametrize("samples_per_frame", [160032])
    def test_bad_samples_per_frame_rate(self, orig_frames: SimpleStream[xr.Dataset], samples_per_frame: int) -> None:
        """Test that ValueError is raised if frame rate is not an integer."""
        with pytest.raises(ValueError, match="samples_per_frame does not yield an integer frame rate"):
            vdif_writer.VDIFFormatter(orig_frames, [{}], station="Me", samples_per_frame=samples_per_frame)

    async def test_success(self, xp, samples_per_frame: int, time_base: Time, time_scale: Fraction) -> None:
        """Test normal usage.

        This includes testing the setting of `invalid_data` bit in the VDIF header.
        The test places a NaN sample in the second VDIF frame for 'v' pol, and
        verifies that in the formatted output data.
        """
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
        # Put a NaN halfway into the second frame, for pol 'v'
        data.data[0, samples_per_frame + samples_per_frame // 2] = xp.nan
        orig = SimpleStream.factory(time_base, time_scale, data, 100)
        # TWO_BIT_1_SIGMA gives compatibility with baseband's encoding
        enc = vdif_writer.VDIFEncode2Bit(orig, samples_per_frame, TWO_BIT_1_SIGMA)
        fmt = vdif_writer.VDIFFormatter(
            enc, [{"pol": "h"}, {"pol": "v"}], station="Me", samples_per_frame=samples_per_frame
        )

        # Write the data to an in-memory file
        fh = io.BytesIO()
        invalid_flags = []
        async for frameset in fmt:
            for frame in frameset:
                invalid_flags.append(bool(baseband.vdif.header.VDIFHeader(frame.header)["invalid_data"]))
                fh.write(frame.header)
                fh.write(frame.payload)
        # `v` is the second configured VDIF thread, so its second frame is
        # the fourth frame received above.
        assert invalid_flags[:4] == [False, False, False, True]
        assert not any(invalid_flags[4:])

        # Read it back and compare to the original data
        fh.seek(0, 0)
        with baseband.vdif.open(fh, "rs") as vdif_data:
            assert vdif_data.sample_rate == float(1 / time_scale) * u.Hz
            assert vdif_data.samples_per_frame == samples_per_frame
            assert vdif_data.bps == 2
            assert vdif_data.shape == data.shape[::-1]
            assert vdif_data.sample_shape == (2,)
            assert (vdif_data.start_time - time_base).sec == pytest.approx(0.002, abs=1e-10)
            # For some reason baseband only exposes this as a key, not an attribute
            assert vdif_data.header0["vdif_version"] == 1
            assert vdif_data.header0.nchan == 1
            assert vdif_data.header0.station == "Me"
            out_data = xr.DataArray(
                vdif_data.read(),
                dims=("time", "pol"),
                coords={"pol": ["h", "v"]},  # Order of `threads`
            )
            # Align out_data to match the axes of data
            out_data = out_data.reindex_like(data).transpose(*data.dims)
            expected_data = data.as_numpy()
            # Confirm where we expect data to be valid (not NaN)
            finite = np.isfinite(expected_data.data)
            # VDIFEncode2Bit marks an entire frame invalid if any sample is NaN.
            # The NaN is in the second frame of the `v` pol; mark it accordingly.
            finite[0, samples_per_frame : 2 * samples_per_frame] = False
            xr.testing.assert_equal(out_data.where(finite), expected_data.where(finite))
