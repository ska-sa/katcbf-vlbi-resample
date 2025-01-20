# Copyright (c) 2025, National Research Foundation (SARAO)

"""Tests for :mod:`katcbf_vlbi_resample.polarisation."""

import numpy as np
import pytest

from katcbf_vlbi_resample.polarisation import parse_spec


class TestParseSpec:
    """Test :func:`katcbf_vlbi_resample.polarisation.parse_spec`."""

    @pytest.mark.parametrize(
        "spec, message",
        [
            ("x,y:x,y:x,y", "polarisation spec 'x,y:x,y:x,y' must contain exactly one colon"),
            ("x,y,x:x,y", "polarisation spec 'x,y,x' must contain exactly one comma"),
            ("x,-x:x,y", "polarisation spec 'x,-x' does not form a basis"),
            ("x,y:x,z", "polarisation 'z' must be x, y, R, L"),
        ],
    )
    def test_bad_spec(self, spec: str, message: str) -> None:
        """Test that appropriate error messages are raised."""
        with pytest.raises(ValueError, match=message):
            parse_spec(spec)

    @pytest.mark.parametrize(
        "spec, expected",
        [
            ("x,y:x,y", np.eye(2)),
            ("x,y:y,x", np.array([[0, 1], [1, 0]])),
            ("-x,y:x,y", np.array([[-1, 0], [0, 1]])),
            ("x,y:x,-y", np.array([[1, 0], [0, -1]])),
            ("x,y:-y,x", np.array([[0, -1], [1, 0]])),
        ],
    )
    def test_simple(self, spec: str, expected: np.ndarray) -> None:
        """Test simple cases just involving transposition and negation."""
        np.testing.assert_equal(parse_spec(spec), expected)
        # With R, L instead of x, y, we're converting from polarisation to
        # linear and back again, so in theory one might not get an exact
        # match.
        spec2 = spec.replace("x", "R").replace("y", "L")
        np.testing.assert_allclose(parse_spec(spec2), expected, rtol=1e-15)

    def test_l2c(self) -> None:
        """Test linear-to-circular conversion."""
        expected = np.array([[1, 1j], [1, -1j]]) * np.sqrt(0.5)
        np.testing.assert_allclose(parse_spec("x,y:R,L"), expected, rtol=1e-15)
        np.testing.assert_allclose(parse_spec("R,L:x,y"), np.linalg.inv(expected), rtol=1e-15)
