# Copyright (c) 2024, National Research Foundation (SARAO)

"""Tests for :mod:`katcbf_vlbi_resample.vdif_writer."""

from fractions import Fraction

import cupy as cp
import pytest
import xarray as xr
from astropy.time import Time
from baseband.vdif.payload import encode_2bit

from katcbf_vlbi_resample.utils import concat_time
from katcbf_vlbi_resample.vdif_writer import VDIFEncode2Bit, _encode_2bit, _encode_2bit_words

from . import SimpleStream


class TestEncode2Bit:
    """Tests for :func:`katcbf_vlbi_resample.vdif_writer._encode_2bit`."""

    def test_random(self, xp) -> None:
        """Test with random data."""
        rng = xp.random.default_rng(seed=2)
        data = rng.uniform(-5.0, 5.0, size=(100000,)).astype(xp.float32)
        expected = encode_2bit(cp.asnumpy(data))
        actual = _encode_2bit(data)
        # Note: values right on the threshold might round differently,
        # but do not currently seem to do so. Changes to cupy's random
        # generator may require adjustments to the test to allow for
        # a little slack.
        xp.testing.assert_array_equal(actual, xp.asarray(expected))


class TestVDIFEncode2Bit:
    """Tests for :class:`katcbf_vlbi_resample.vdif_writer.VDIFEncode2Bit."""

    @pytest.fixture
    def input_data(self, xp) -> xr.DataArray:
        """Input data, as a single chunk."""
        return xr.DataArray(
            xp.tile(xp.arange(-2.5, 3, dtype=xp.float32), 100),
            dims=("time",),
            attrs={"time_bias": Fraction(137)},
        )

    @pytest.fixture
    def orig(self, input_data: xr.DataArray, time_base: Time, time_scale: Fraction) -> SimpleStream:
        """Input stream."""
        return SimpleStream(time_base, time_scale, input_data, 10)

    def test_bad_samples_per_frame_word_align(self, orig: SimpleStream) -> None:
        """Test that `samples_per_frame` not a multiple of word size raises :exc:`ValueError`."""
        with pytest.raises(ValueError, match="samples_per_frame must be a multiple of 16"):
            VDIFEncode2Bit(orig, 12345)

    def test_bad_samples_per_frame_rate(self, orig: SimpleStream) -> None:
        """Test that ValueError is raised if frame rate is not an integer."""
        with pytest.raises(ValueError, match="samples_per_frame does not yield an integer frame rate"):
            VDIFEncode2Bit(orig, 160032)

    def test_success(self, xp, orig: SimpleStream, input_data: xr.DataArray) -> None:
        """Test normal usage."""
        enc = VDIFEncode2Bit(orig, 80)
        assert enc.time_base == orig.time_base
        assert enc.time_scale == orig.time_scale * VDIFEncode2Bit.SAMPLES_PER_WORD
        assert enc.is_cupy == orig.is_cupy
        # VDIFEncode2Bit should be aligning things to frame boundaries. The
        # time_base is already on a frame boundary, so frame boundaries occur
        # when time_bias is a multiple of the frame size.
        chunks = list(enc)
        out_data = concat_time(chunks)
        assert out_data.attrs["time_bias"] == Fraction(160, VDIFEncode2Bit.SAMPLES_PER_WORD)
        assert out_data.sizes["time"] == 560 // VDIFEncode2Bit.SAMPLES_PER_WORD
        used_input_data = input_data.isel(time=xp.s_[23:583])
        xp.testing.assert_array_equal(out_data.data, _encode_2bit_words(used_input_data.data))
