# Copyright (c) 2024, National Research Foundation (SARAO)

"""Tests for :mod:`katcbf_vlbi_resample.power."""

from fractions import Fraction

import xarray as xr
from astropy.time import Time

from katcbf_vlbi_resample.power import NormalisePower
from katcbf_vlbi_resample.utils import concat_time

from . import SimpleStream


class TestNormalisePower:
    """Test :class:`.NormalisePower`."""

    def test(self, xp, time_base: Time, time_scale: Fraction) -> None:
        """Test :class:`.NormalisePower`."""
        rng = xp.random.default_rng(seed=1)
        data = xr.DataArray(
            rng.standard_normal(size=(2, 10000)) * 4.5,  # cupy doesn't implement Generator.normal
            dims=("pol", "time"),
            attrs={"time_bias": Fraction(123, 456)},
        )
        data[0] *= 2  # Give the pols different scales
        stream = SimpleStream(time_base, time_scale, data, 10)
        norm = NormalisePower(stream, 1.5)
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
