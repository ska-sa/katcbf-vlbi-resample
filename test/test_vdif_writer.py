# Copyright (c) 2024, National Research Foundation (SARAO)

"""Tests for :mod:`katcbf_vlbi_resample.vdif_writer."""

import cupy as cp
from baseband.vdif.payload import encode_2bit

from katcbf_vlbi_resample.vdif_writer import _encode_2bit


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
