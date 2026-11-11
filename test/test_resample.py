################################################################################
# Copyright (c) 2024-2026, National Research Foundation (SARAO)
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

"""Tests for :mod:`katcbf_vlbi_resample.resample."""

from fractions import Fraction

import pytest
import xarray as xr
from astropy.time import Time

from katcbf_vlbi_resample.parameters import ResampleParameters
from katcbf_vlbi_resample.resample import IFFT, ClipTime, Resample
from katcbf_vlbi_resample.utils import concat_time

from . import SimpleStream, complex_random


class TestClipTime:
    """Test :class:`.ClipTime`."""

    @pytest.fixture
    def orig(self, xp, time_base: Time, time_scale: Fraction) -> SimpleStream[xr.DataArray]:
        """Input stream."""
        data = xr.DataArray(
            xp.arange(1000, 50000, 100),
            dims=("time",),
            attrs={"time_bias": 50},
        )
        return SimpleStream.factory(time_base, time_scale, data, 10)

    @pytest.mark.parametrize(
        "start,stop,time_bias,n",
        [
            (0, 10000, 50, 490),  # Start and stop span the whole data range
            (None, None, 50, 490),  # No start or stop
            (170, 290, 170, 120),  # Start and stop aligned to chunks
            (173, 298, 173, 125),  # Stop and stop not aligned to chunks
        ],
    )
    async def test_overlap(
        self,
        xp,
        orig: SimpleStream[xr.DataArray],
        start: int | None,
        stop: int | None,
        time_bias: int,
        n: int,
    ) -> None:
        """Test where selected range overlaps the data."""
        clip = ClipTime(orig, start, stop)
        assert clip.time_base == orig.time_base
        assert clip.time_scale == orig.time_scale
        chunks = [chunk async for chunk in clip]
        next_time_bias = time_bias
        for chunk in chunks:
            assert chunk.sizes["time"]  # Empty chunks should be skipped
            assert chunk.attrs["time_bias"] == next_time_bias
            next_time_bias += chunk.sizes["time"]
        out = xr.concat(chunks, dim="time")
        assert out.sizes["time"] == n
        xp.testing.assert_array_equal(out.data, xp.arange(time_bias, time_bias + n) * 100 - 4000)

    @pytest.mark.parametrize(
        "start,stop",
        [(10, 40), (10, 50), (None, 40), (540, 1000), (550, 1000), (540, None)],
    )
    async def test_no_overlap(self, orig: SimpleStream[xr.DataArray], start: int | None, stop: int | None) -> None:
        """Test where selected range does not overlap the data."""
        clip = ClipTime(orig, start, stop)
        assert clip.time_base == orig.time_base
        assert clip.time_scale == orig.time_scale
        chunks = [chunk async for chunk in clip]
        assert not chunks

    def test_absolute_time(self, orig: SimpleStream[xr.DataArray]) -> None:
        """Test absolute times for start and stop."""
        clip = ClipTime(orig, Time("2024-07-20T12:00:02.25", scale="utc"), Time("2024-07-20T12:00:02.75", scale="utc"))
        assert clip._start == 320000
        assert clip._stop == 400000


class TestIFFT:
    """Test :class:`.IFFT`."""

    async def test(self, xp, time_base: Time, time_scale: Fraction) -> None:
        """Test normal usage."""
        rng = xp.random.default_rng(seed=1)
        channels = 32768
        spectra = 123
        time_data = complex_random(lambda: rng.uniform(-1.0, 1.0, (spectra, channels)))
        freq_data = xp.fft.fft(time_data, axis=1, norm="ortho")
        freq_data = xp.fft.fftshift(freq_data, axes=1)
        freq_data_xr = xr.DataArray(
            freq_data,
            dims=("time", "channel"),
            attrs={"time_bias": 4321},
        )
        stream = SimpleStream.factory(time_base, time_scale, freq_data_xr, 10)
        ifft = IFFT(stream)
        assert ifft.channels is None
        assert ifft.time_base == stream.time_base
        assert ifft.time_scale == stream.time_scale / channels
        out = concat_time([chunk async for chunk in ifft])
        assert out.attrs["time_bias"] == freq_data_xr.attrs["time_bias"] * channels
        xp.testing.assert_allclose(time_data.ravel(), out.data)

    def test_no_channels(self, xp, time_base: Time, time_scale: Fraction) -> None:
        """Test error handling when the input stream is not channelised."""
        data = xr.DataArray(xp.zeros(100), dims=("time",), attrs={"time_bias": 0})
        stream = SimpleStream.factory(time_base, time_scale, data)
        with pytest.raises(TypeError):
            IFFT(stream)


class TestResample:
    """Test :class:`.Resample`."""

    @pytest.fixture(params=[0.0, 8e6])
    def mixer_frequency(self, request: pytest.FixtureRequest) -> float:
        """Difference between input and output centre frequencies.

        This fixture is parametrised to test both the fast path when
        the mixer frequency is zero as well as the more general case.
        """
        return request.param

    @pytest.fixture
    def output_bandwidth(self) -> float:
        """Output bandwidth."""
        return 64e6

    @pytest.fixture
    def resample_params(self) -> ResampleParameters:
        """Resampling parameters fixture."""
        return ResampleParameters(
            fir_taps=7201,
            hilbert_taps=201,
            passband=0.95,
        )

    @pytest.fixture
    def time_scale(self) -> Fraction:
        """Time scale for input stream."""
        return 1 / Fraction(107e6)

    @pytest.fixture
    def time_size(self) -> int:
        """Time samples in an input stream."""
        return 54321

    @pytest.mark.parametrize("chunk_size", [100, 20000])
    async def test_chunk_consistency(
        self,
        xp,
        output_bandwidth: float,
        mixer_frequency: float,
        resample_params: ResampleParameters,
        time_base: Time,
        time_scale: Fraction,
        chunk_size: int,
        time_size: int,
    ) -> None:
        """Verify that results are not affected by chunk boundaries."""
        rng = xp.random.default_rng(seed=1)
        data = xr.DataArray(
            complex_random(lambda: rng.uniform(-1.0, 1.0, size=(2, time_size)).astype(xp.float32)),
            dims=("pol", "time"),
            coords={"pol": ["h", "v"]},
            attrs={"time_bias": 12345},
        )
        orig1 = SimpleStream.factory(time_base, time_scale, data)
        orig2 = SimpleStream.factory(time_base, time_scale, data, chunk_size)
        resample1 = Resample(output_bandwidth, mixer_frequency, resample_params, orig1)
        resample2 = Resample(output_bandwidth, mixer_frequency, resample_params, orig2)
        out1 = [chunk async for chunk in resample1]
        out2 = [chunk async for chunk in resample2]
        out1c = xr.concat(out1, dim="time")
        out2c = xr.concat(out2, dim="time")
        # TODO: the cupy version seems to need a much higher atol to pass.
        xr.testing.assert_allclose(out1c, out2c, atol=1e-5)

    async def test_gap_filled_with_nans(
        self,
        xp,
        output_bandwidth: float,
        mixer_frequency: float,
        resample_params: ResampleParameters,
        time_base: Time,
        time_scale: Fraction,
        time_size: int,
    ) -> None:
        """Verify that gaps in the input are filled with NaNs."""
        rng = xp.random.default_rng(seed=1)
        data = xr.DataArray(
            complex_random(lambda: rng.uniform(-1.0, 1.0, size=(2, time_size)).astype(xp.float32)),
            dims=("pol", "time"),
            coords={"pol": ["h", "v"]},
            attrs={"time_bias": 12345},
        )
        # Split the data into two chunks then create a gap between them
        gap_start = time_size // 3
        chunk_sizes = [gap_start, time_size - gap_start]
        orig_stream = SimpleStream.factory(time_base, time_scale, data, chunk_sizes)
        gap_size = 1000
        chunks = list(orig_stream.chunks)
        second_chunk = chunks[1]
        # Shift the second chunk forward in time to create the gap
        second_chunk.attrs = {
            **second_chunk.attrs,
            "time_bias": second_chunk.attrs["time_bias"] + gap_size,
        }

        filler = xr.DataArray(
            xp.full((2, gap_size), xp.nan + 1j * xp.nan, dtype=data.dtype),
            dims=data.dims,
            coords=data.coords,
        )
        reference_data: xr.DataArray = xr.concat(
            [
                data.isel(time=slice(0, gap_start)),
                filler,
                data.isel(time=slice(gap_start, None)),
            ],
            dim="time",
        ).assign_attrs(dict(data.attrs))
        # The reference_stream chunk sizes need to account for the "left-padding" of the
        # gap in data introduced.
        reference_stream = SimpleStream.factory(
            time_base, time_scale, reference_data, [gap_start, gap_size + time_size - gap_start]
        )

        orig_resample = Resample(output_bandwidth, mixer_frequency, resample_params, orig_stream)
        reference_resample = Resample(output_bandwidth, mixer_frequency, resample_params, reference_stream)
        orig_out = concat_time([chunk async for chunk in orig_resample])
        reference_out = concat_time([chunk async for chunk in reference_resample])

        assert orig_out.attrs["time_bias"] == reference_out.attrs["time_bias"]
        assert orig_out.sizes["time"] == reference_out.sizes["time"]
        # NOTE: Cannot do DataArray.isnull because cupy requires copying to host first
        xp.testing.assert_array_equal(xp.isnan(orig_out.data), xp.isnan(reference_out.data))
        # Mask the two non-NaN locations to ensure neither `orig_out` nor `reference_out`
        # have been corrupted by the NaNs in the input.
        finite = xp.isfinite(orig_out.data) & xp.isfinite(reference_out.data)
        xp.testing.assert_allclose(orig_out.data[finite], reference_out.data[finite], atol=1e-5)

    async def test_group_delay(
        self,
        xp,
        output_bandwidth: float,
        mixer_frequency: float,
        resample_params: ResampleParameters,
        time_base: Time,
        time_scale: Fraction,
    ) -> None:
        """Check that timestamp attributes correctly yield zero group delay."""
        # Frequencies relative to the centre frequency
        freqs = [-28e6, -18e6, 17e6, 22e6]
        n = 100000
        attrs = {"time_bias": 12345}
        t = (xp.arange(n) + float(attrs["time_bias"])) * float(time_scale)
        tones = xp.stack([xp.exp(2j * xp.pi * f * t) for f in freqs])
        data = xr.DataArray(
            tones,
            dims=("freq", "time"),
            coords={"freq": freqs},
            attrs=attrs,
        )
        orig = SimpleStream.factory(time_base, time_scale, data)
        resample = Resample(output_bandwidth, mixer_frequency, resample_params, orig)
        out = [chunk async for chunk in resample]
        assert len(out) == 1
        for f in freqs:
            res = out[0].sel(freq=f)
            if f < -mixer_frequency:
                res = res.sel(sideband="lsb")
            else:
                res = res.sel(sideband="usb")
            # Regenerate the expected tone at the adjusted sampling rate and using
            # the updated timestamp information
            t = (xp.arange(res.sizes["time"]) + float(res.attrs["time_bias"])) * float(resample.time_scale)
            tone = xp.exp(2j * xp.pi * (f + mixer_frequency) * t)
            # Correlate to get the phase
            phase = float(xp.angle(xp.vdot(tone, res.data)))
            assert phase == pytest.approx(0.0, abs=1e-7)
