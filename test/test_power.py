# Copyright (c) 2024, National Research Foundation (SARAO)

"""Tests for :mod:`katcbf_vlbi_resample.power."""

from fractions import Fraction

import numpy as np
import pytest
import xarray as xr
from astropy.time import Time

from katcbf_vlbi_resample.power import MeasurePower, NormalisePower, RecordPower
from katcbf_vlbi_resample.utils import concat_time, is_cupy, isel_time

from . import SimpleStream


class StoreRms(RecordPower):
    """Subclass that stores RMS history in a list."""

    _MAX_HISTORY = 5  # Override base class value to test flushing

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.history: list[tuple[int, int, xr.DataArray]] = []

    def record_rms(self, start: int, length: int, rms: xr.DataArray) -> None:  # noqa: D102
        assert not is_cupy(rms)
        self.history.append((start, length, rms))


class TestMeasurePower:
    """Test :class:`.MeasurePower`."""

    def test(self, xp, time_base: Time, time_scale: Fraction) -> None:
        """Test :class:`.NormalisePower`."""
        rng = xp.random.default_rng(seed=1)
        data = xr.DataArray(
            rng.standard_normal(size=(2, 100000)) * 4.5,  # cupy doesn't implement Generator.normal
            dims=("pol", "time"),
            coords={"pol": ["h", "v"]},
            attrs={"time_bias": 123},
            name="data",
        )
        data[0] *= 2  # Give the pols different scales
        data[1, 10000:30000] *= 3  # Introduce variation in time
        stream = SimpleStream.factory(time_base, time_scale, data, 10000)
        norm = MeasurePower(stream)
        assert norm.time_base == time_base
        assert norm.time_scale == time_scale
        assert norm.channels is None
        chunks = list(norm)
        out = concat_time([chunk["data"] for chunk in chunks])
        out_rms = xr.concat([chunk["rms"] for chunk in chunks], dim="time").T
        xr.testing.assert_identical(out, data)
        expected_rms = xr.DataArray(
            xp.full((2, 10), 4.5),
            dims=("pol", "time"),
            coords={"pol": ["h", "v"]},
            name="rms",
        )
        expected_rms[0] *= 2
        expected_rms[1, 1:3] *= 3
        xr.testing.assert_allclose(out_rms, expected_rms, rtol=0.05)
        # Replace the data so that we can check the metadata more strictly
        expected_rms = expected_rms.copy(data=out_rms)
        xr.testing.assert_identical(out_rms, expected_rms)


@pytest.fixture
def orig_data(xp) -> xr.DataArray:
    """Test data samples."""
    return xr.DataArray(
        xp.arange(12, dtype=xp.float32).reshape(2, 6),
        dims=("pol", "time"),
        coords={"pol": ["h", "v"]},
        attrs={"time_bias": 123},
    )


@pytest.fixture
def orig_rms(xp) -> xr.DataArray:
    """RMS values.

    These are not actual RMS values from orig_data. They've chosen to be
    make it easy to compute the expected outputs.
    """
    return xr.DataArray(
        xp.array(
            [
                [4.0, 8.0],
                [1.0, 1.0],
                [3.0, 1.5],
            ],
            dtype=xp.float32,
        ),
        dims=("chunk", "pol"),
        coords={"pol": ["h", "v"]},
    )


@pytest.fixture
def orig(
    xp, time_base: Time, time_scale: Fraction, orig_data: xr.DataArray, orig_rms: xr.DataArray
) -> SimpleStream[xr.Dataset]:
    """Input stream."""
    chunks = [
        xr.Dataset(
            {"data": isel_time(orig_data, xp.s_[i * 2 : (i + 1) * 2]).copy(), "rms": orig_rms.isel(chunk=i).copy()}
        )
        for i in range(orig_rms.sizes["chunk"])
    ]
    return SimpleStream(
        time_base=time_base,
        time_scale=time_scale,
        channels=None,
        is_cupy=is_cupy(orig_data),
        chunks=chunks,
    )


class TestRecordPower:
    """Test :class:`.RecordPower`."""

    def test(self, xp, orig_rms: xr.DataArray, orig: SimpleStream[xr.Dataset]) -> None:  # noqa: D102
        output = []
        recorder = StoreRms(orig)
        for chunk in recorder:
            output.append(chunk["rms"])
        output_array = xr.concat(output, dim="chunk")
        orig_rms.name = "rms"
        xr.testing.assert_identical(output_array, orig_rms)

        assert len(recorder.history) == len(output)
        for i, (start, length, rms) in enumerate(recorder.history):
            assert start == 123 + 2 * i
            assert length == 2
        history_array = xr.concat([entry[2] for entry in recorder.history], dim="chunk")
        xr.testing.assert_identical(history_array, orig_rms.as_numpy())


class TestNormalisePower:
    """Test :class:`.NormalisePower`."""

    def _test_stream_attributes(self, norm: NormalisePower, orig: SimpleStream) -> None:
        assert norm.time_base == orig.time_base
        assert norm.time_scale == orig.time_scale
        assert norm.channels == orig.channels
        assert norm.is_cupy == orig.is_cupy

    def test_auto(self, xp, orig: SimpleStream[xr.Dataset]) -> None:
        """Test :class:`.NormalisePower` with 'auto' normalisation."""
        norm = NormalisePower(orig, scale=1.5, power="auto")
        self._test_stream_attributes(norm, orig)
        data = concat_time(list(norm))
        assert is_cupy(data) == norm.is_cupy

        expected = xr.DataArray(
            xp.array(
                [
                    [0.0, 0.375, 3.0, 4.5, 2.0, 2.5],
                    [1.125, 1.3125, 12.0, 13.5, 10.0, 11.0],
                ],
                dtype=xp.float32,
            ),
            dims=("pol", "time"),
            coords={"pol": ["h", "v"]},
            attrs={"time_bias": 123},
            name="data",
        )
        xr.testing.assert_identical(data, expected)

    def test_first(self, xp, orig: SimpleStream[xr.Dataset]) -> None:
        """Test :class:`.NormalisePower` with 'first' normalisation."""
        norm = NormalisePower(orig, scale=1.5, power="first")
        self._test_stream_attributes(norm, orig)
        data = concat_time(list(norm))
        assert is_cupy(data) == norm.is_cupy

        expected = xr.DataArray(
            xp.array(
                [
                    [0.0, 0.375, 0.75, 1.125, 1.5, 1.875],
                    [1.125, 1.3125, 1.5, 1.6875, 1.875, 2.0625],
                ],
                dtype=xp.float32,
            ),
            dims=("pol", "time"),
            coords={"pol": ["h", "v"]},
            attrs={"time_bias": 123},
            name="data",
        )
        xr.testing.assert_identical(data, expected)

    def test_fixed(self, xp, orig: SimpleStream[xr.Dataset]) -> None:
        """Test :class:`.NormalisePower` with user-provided normalisation."""
        power = xr.DataArray(
            np.array([9.0, 4.0]),  # Note: np not xp because it's provided by user
            dims=("pol",),
            coords={"pol": ["v", "h"]},  # Coordinates swapped for more thorough testing
        )
        norm = NormalisePower(orig, scale=1.5, power=power)
        self._test_stream_attributes(norm, orig)
        data = concat_time(list(norm))
        assert is_cupy(data) == norm.is_cupy

        expected = xr.DataArray(
            xp.array(
                [
                    [0.0, 0.75, 1.5, 2.25, 3.0, 3.75],
                    [3.0, 3.5, 4.0, 4.5, 5.0, 5.5],
                ],
                dtype=xp.float32,
            ),
            dims=("pol", "time"),
            coords={"pol": ["h", "v"]},
            attrs={"time_bias": 123},
            name="data",
        )
        xr.testing.assert_identical(data, expected)
