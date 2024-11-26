# Copyright (c) 2024, National Research Foundation (SARAO)

"""Tests for :mod:`katcbf_vlbi_resample.power."""

from fractions import Fraction

import pytest
import xarray as xr
from astropy.time import Time

from katcbf_vlbi_resample.power import NormalisePower
from katcbf_vlbi_resample.utils import concat_time, is_cupy

from . import SimpleStream


class StoreRms(NormalisePower):
    """Subclass that stores RMS history in a list."""

    _MAX_HISTORY = 5  # Override base class value to test flushing

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.history: list[tuple[int, int, xr.DataArray]] = []

    def record_rms(self, start: int, length: int, rms: xr.DataArray) -> None:  # noqa: D102
        assert not is_cupy(rms)
        self.history.append((start, length, rms))
        super().record_rms(start, length, rms)


class TestNormalisePower:
    """Test :class:`.NormalisePower`."""

    def test(self, xp, time_base: Time, time_scale: Fraction) -> None:
        """Test :class:`.NormalisePower`."""
        rng = xp.random.default_rng(seed=1)
        data = xr.DataArray(
            rng.standard_normal(size=(2, 100000)) * 4.5,  # cupy doesn't implement Generator.normal
            dims=("pol", "time"),
            coords={"pol": ["h", "v"]},
            attrs={"time_bias": 123},
        )
        data[0] *= 2  # Give the pols different scales
        stream = SimpleStream(time_base, time_scale, data, 10000)
        norm = StoreRms(stream, 1.5)
        assert norm.time_base == time_base
        assert norm.time_scale == time_scale
        assert norm.channels is None
        out = concat_time(list(norm))
        assert out.attrs["time_bias"] == data.attrs["time_bias"]
        # Note: NormalisePower normalises rms not std, which are different
        # if the mean is non-zero. We've chosen a distribution with mean
        # zero, but there will be random variation, so we don't expect an
        # exact match.
        std = out.std(dim="time")
        xp.testing.assert_allclose(std.data, 1.5, rtol=0.02)
        assert len(norm.history) == 10
        for i, (start, length, rms) in enumerate(norm.history):
            assert start == 123 + 10000 * i
            assert length == 10000
            assert rms[0] == pytest.approx(9.0, rel=0.02)
            assert rms[1] == pytest.approx(4.5, rel=0.02)
            # Check the metadata, by filling in identical data
            expected = xr.DataArray(
                rms.data,
                dims=("pol",),
                coords={"pol": ["h", "v"]},
            )
            print(rms)
            print(expected)
            xr.testing.assert_identical(rms, expected)
