# Copyright (c) 2024, National Research Foundation (SARAO)

"""Tests for :mod:`katcbf_vlbi_resample.resample."""

from collections.abc import Iterator
from fractions import Fraction

import numpy as np
import pytest
import xarray as xr
from astropy.time import Time

from katcbf_vlbi_resample.parameters import ResampleParameters, StreamParameters
from katcbf_vlbi_resample.resample import ClipTime, Resample


class SimpleStream:
    """Stream that holds its data in memory."""

    def __init__(
        self, time_base: Time, time_scale: Fraction, data: xr.DataArray, chunk_size: int | None = None
    ) -> None:
        self.time_base = time_base
        self.time_scale = time_scale
        self.channels = data.sizes.get("channels")
        if chunk_size is None:
            self.chunks = [data]
        else:
            self.chunks = []
            for start in range(0, data.sizes["time"], chunk_size):
                stop = min(start + chunk_size, data.sizes["time"])
                chunk = data.isel(time=np.s_[start:stop])
                chunk.attrs["time_bias"] += start
                self.chunks.append(chunk)

    def __iter__(self) -> Iterator[xr.DataArray]:
        return iter(self.chunks)


class TestClipTime:
    """Test :class:`.ClipTime`."""

    @pytest.fixture
    def data(self) -> xr.DataArray:
        """Input data, as a single chunk."""
        return xr.DataArray(
            np.arange(1000, 50000, 100),
            dims=("time",),
            attrs={"time_bias": Fraction(50)},
        )

    @pytest.mark.parametrize(
        "start,stop,time_bias,n",
        [
            (0, 10000, 50, 490),  # Start and stop span the whole data range
            (170, 290, 170, 120),  # Start and stop aligned to chunks
            (173, 298, 173, 125),  # Stop and stop not aligned to chunks
        ],
    )
    def test_overlap(self, data: xr.DataArray, start: int, stop: int, time_bias: int, n: int) -> None:
        """Test where selected range overlaps the data."""
        time_base = Time("2024-07-20T12:00:00", scale="utc")
        time_scale = Fraction(1, 12345)
        orig = SimpleStream(time_base, time_scale, data, 10)
        clip = ClipTime(orig, start, stop)
        assert clip.time_base == orig.time_base
        assert clip.time_scale == orig.time_scale
        chunks = list(clip)
        next_time_bias = time_bias
        for chunk in chunks:
            assert chunk.sizes["time"]  # Empty chunks should be skipped
            assert chunk.attrs["time_bias"] == next_time_bias
            next_time_bias += chunk.sizes["time"]
        out = xr.concat(chunks, dim="time")
        assert out.sizes["time"] == n
        np.testing.assert_equal(out.data, np.arange(time_bias, time_bias + n) * 100 - 4000)


class TestResample:
    """Test :class:`.Resample`."""

    @pytest.fixture
    def input_params(self) -> StreamParameters:
        """Input parameters fixture."""
        return StreamParameters(
            bandwidth=107e6,
            center_freq=428e6,
        )

    @pytest.fixture
    def output_params(self) -> StreamParameters:
        """Output parameters fixture."""
        return StreamParameters(
            bandwidth=64e6,
            center_freq=420e6,
        )

    @pytest.fixture
    def resample_params(self) -> ResampleParameters:
        """Resampling parameters fixture."""
        return ResampleParameters(
            fir_taps=7201,
            hilbert_taps=201,
            passband=0.95,
        )

    @pytest.fixture
    def time_base(self) -> Time:
        """Time base for input stream."""
        return Time("2024-07-20T12:00:00", scale="utc")

    @pytest.fixture
    def time_scale(self, input_params: StreamParameters):
        """Time scale for input stream."""
        return 1 / Fraction(input_params.bandwidth)

    def test_chunk_consistency(
        self,
        input_params: StreamParameters,
        output_params: StreamParameters,
        resample_params: ResampleParameters,
        time_base: Time,
        time_scale: Fraction,
    ) -> None:
        """Verify that results are not affected by chunk boundaries."""
        rng = np.random.default_rng(seed=1)
        data = xr.DataArray(
            rng.uniform(-1.0, 1.0, size=(2, 1048576)) + 1j * rng.uniform(-1.0, 1.0, size=(2, 1048576)),
            dims=("pol", "time"),
            coords={"pol": ["h", "v"]},
            attrs={"time_bias": Fraction(12345)},
        )
        orig1 = SimpleStream(time_base, time_scale, data)
        orig2 = SimpleStream(time_base, time_scale, data, 50000)
        resample1 = Resample(input_params, output_params, resample_params, orig1)
        resample2 = Resample(input_params, output_params, resample_params, orig2)
        out1 = list(resample1)
        out2 = list(resample2)
        out1c = xr.concat(out1, dim="time")
        out2c = xr.concat(out2, dim="time")
        xr.testing.assert_allclose(out1c, out2c)

    def test_group_delay(
        self,
        input_params: StreamParameters,
        output_params: StreamParameters,
        resample_params: ResampleParameters,
        time_base: Time,
        time_scale: Fraction,
    ) -> None:
        """Check that timestamp attributes correctly yield zero group delay."""
        freqs = [400e6, 410e6, 445e6, 450e6]
        n = 100000
        attrs = {"time_bias": Fraction(12345)}
        t = (np.arange(n) + float(attrs["time_bias"])) * float(time_scale)
        tones = np.stack([np.exp(2j * np.pi * (f - input_params.center_freq) * t) for f in freqs])
        data = xr.DataArray(
            tones,
            dims=("freq", "time"),
            coords={"freq": freqs},
            attrs=attrs,
        )
        orig = SimpleStream(time_base, time_scale, data)
        resample = Resample(input_params, output_params, resample_params, orig)
        out = list(resample)
        assert len(out) == 1
        for f in freqs:
            res = out[0].sel(freq=f)
            if f < output_params.center_freq:
                res = res.sel(sideband="lsb")
            else:
                res = res.sel(sideband="usb")
            # Regenerate the expected tone at the adjusted sampling rate and using
            # the updated timestamp information
            t = (np.arange(res.sizes["time"]) + float(res.attrs["time_bias"])) * float(resample.time_scale)
            tone = np.exp(2j * np.pi * (f - output_params.center_freq) * t)
            # Correlate to get the phase
            phase = np.angle(np.vdot(tone, res.as_numpy()))
            assert phase == pytest.approx(0.0, abs=1e-7)
