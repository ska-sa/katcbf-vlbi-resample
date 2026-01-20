################################################################################
# Copyright (c) 2025, National Research Foundation (SARAO)
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

"""Tests for :mod:`katcbf_vlbi_resample.polarisation."""

from fractions import Fraction

import numpy as np
import pytest
import xarray as xr
from astropy.time import Time

from katcbf_vlbi_resample.polarisation import ConvertPolarisation, parse_spec, to_linear
from katcbf_vlbi_resample.utils import concat_time, is_cupy

from . import SimpleStream


class TestParseSpec:
    """Test :func:`katcbf_vlbi_resample.polarisation.parse_spec`."""

    @pytest.mark.parametrize(
        "spec, message",
        [
            ("x,y:x,y:x,y", "polarisation spec 'x,y:x,y:x,y' must contain exactly one colon"),
            ("x,y,x:x,y", "polarisation spec 'x,y,x' must contain exactly one comma"),
            ("x,-x:x,y", "polarisations x,-x do not form a basis"),
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


def test_to_linear_bad_length() -> None:
    """Test error handling in :func:`katcbf_vlbi_resample.polarisation.to_linear."""
    # The normal functionality is indirectly tested via TestParseSpec above.
    with pytest.raises(ValueError, match="must contain exactly two elements"):
        to_linear(["x", "y", "R"])


class TestConvertPolarisation:
    """Test :class:`katcbf_vlbi_resample.polarisation.ConvertPolarisation`."""

    def _make_orig_data(self, xp, in_pol_labels: list[str]) -> xr.DataArray:
        return xr.DataArray(
            xp.array(
                [
                    [1 + 2j, 3 + 5j, 8 - 11j, 1.5, -2.5],
                    [-7j, 1 + 9.5j, 10j, 0.0, 1.0],
                ],
                np.complex64,
            ),
            dims=("pol", "time"),
            coords={"pol": in_pol_labels},
            attrs={"time_bias": 123},
        )

    @pytest.mark.parametrize("in_pol_labels", [["pol0", "pol1"], ["hello", "world"]])
    @pytest.mark.parametrize("out_pol_labels", [["pol0", "pol1"], ["world", "hello"]])
    async def test(
        self, xp, time_base: Time, time_scale: Fraction, in_pol_labels: list[str], out_pol_labels: list[str]
    ) -> None:
        """Test :class:`katcbf_vlbi_resample.polarisation.ConvertPolarisation`."""
        orig_data = self._make_orig_data(xp, in_pol_labels)
        orig = SimpleStream.factory(time_base, time_scale, orig_data, chunk_size=3)
        # This isn't a realistic polarisation basis matrix, but it makes it
        # easy to compute expected values.
        matrix = np.array([[0.0, 2.0], [1.0, -1.0]], np.complex64)
        out = ConvertPolarisation(orig, matrix, in_pol_labels=in_pol_labels, out_pol_labels=out_pol_labels)

        assert out.is_cupy == orig.is_cupy
        assert out.time_base == orig.time_base
        assert out.time_scale == orig.time_scale
        assert out.channels == orig.channels
        data = concat_time([chunk async for chunk in out])
        assert is_cupy(data) == out.is_cupy
        expected = xr.DataArray(
            xp.array(
                [
                    [-14j, 2 + 19j, 20j, 0.0, 2.0],
                    [1 + 9j, 2 - 4.5j, 8 - 21j, 1.5, -3.5],
                ],
                np.complex64,
            ),
            dims=("pol", "time"),
            coords={"pol": out_pol_labels},
            attrs={"time_bias": 123},
        )
        xr.testing.assert_identical(data, expected)

    async def test_pol_label_mismatch(self, xp, time_base: Time, time_scale: Fraction) -> None:
        """Test that mismatch of polarisation labels causes an error."""
        # Reverse the order of the labels on the input
        orig_data = self._make_orig_data(xp, ["pol1", "pol0"])
        orig = SimpleStream.factory(time_base, time_scale, orig_data, chunk_size=3)
        matrix = np.eye(2, dtype=np.complex64)
        out = ConvertPolarisation(orig, matrix)
        with pytest.raises(RuntimeError):
            async for chunk in out:
                pass
