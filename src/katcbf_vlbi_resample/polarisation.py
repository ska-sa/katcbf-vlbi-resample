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

"""Convert polarised data between bases.

Note that the equations in this module require that the electric field sign
convention is such that phase *increases* with time at a given position in
space. That is valid for MeerKAT for *not* for KAT-7.
"""

import math
import re
from collections.abc import Iterable

import cupy as cp
import numpy as np
import xarray as xr

from .stream import ChunkwiseStream, Stream

# Specify each polarisation in a linear x/y basis.
# Refer to Hamaker and Bregman Paper III for the sign conventions.
# We assume a plus sign in equation 2 i.e. phase increases with
# time at a given point in space.
_POLARISATIONS = {
    "x": [1.0, 0.0],
    "y": [0.0, 1.0],
    "R": [math.sqrt(0.5), -1j * math.sqrt(0.5)],
    "L": [math.sqrt(0.5), +1j * math.sqrt(0.5)],
}


def _parse_pol(spec: str) -> np.ndarray:
    """Parse a single polarisation in the :option:`--polarisation` command-line option.

    The return value is a vector giving the coordinates of a signal of
    this polarisation in linear x, y coordinates.

    Raises
    ------
    ValueError
        if the argument does not match the expected format.
    """
    if not re.fullmatch("[-+]?[xyRL]", spec):
        raise ValueError(f"polarisation {spec!r} must be x, y, R, L, optionally with a sign prefix")
    v = np.array(_POLARISATIONS[spec[-1]], dtype=np.complex128)
    if spec[0] == "-":
        v = -v
    return v


def _parse_half_spec(spec: str) -> np.ndarray:
    """Parse one side of the :option:`!--polarisation` command-line option.

    The return value is a Jones matrix which converts from the given
    coordinate system to a linear coordinate system.

    Raises
    ------
    ValueError
        if the argument does not match the expected format.
    """
    parts = spec.split(",")
    if len(parts) != 2:
        raise ValueError(f"polarisation spec {spec!r} must contain exactly one comma")
    m = np.column_stack([_parse_pol(part) for part in parts])
    if np.linalg.matrix_rank(m) < 2:
        raise ValueError(f"polarisation spec {spec!r} does not form a basis")
    return m


def parse_spec(spec: str) -> np.ndarray:
    """Parse the :option:`!--polarisation` command-line option.

    The return value is a Jones matrix which is multiplied by
    the electric field vector to get the output vector.

    Raises
    ------
    ValueError
        if the argument does not match the expected format.
    """
    parts = spec.split(":")
    if len(parts) != 2:
        raise ValueError(f"polarisation spec {spec!r} must contain exactly one colon")
    in_pols = _parse_half_spec(parts[0])
    out_pols = _parse_half_spec(parts[1])
    return np.linalg.inv(out_pols) @ in_pols


class ConvertPolarisation(ChunkwiseStream[xr.DataArray, xr.DataArray]):
    """Convert polarisation basis of a stream.

    Parameters
    ----------
    input_data
        Input data stream. Each chunk must contain an axis called `pol` with
        coordinates given by `in_pol_labels`, and must not contain an axis
        called `in_pol`.
    matrix
        Jones matrix to convert from input to output polarisation basis.
    in_pol_labels
        Coordinates used for the input polarisations (must be a list of 2 elements)
    out_pol_labels
        Coordinates that will be used by the polarisations on output.
    """

    def __init__(
        self,
        input_data: Stream[xr.DataArray],
        matrix: np.ndarray,
        in_pol_labels: Iterable[str] = ("pol0", "pol1"),
        out_pol_labels: Iterable[str] = ("pol0", "pol1"),
    ) -> None:
        super().__init__(input_data)
        self.in_pol_labels = list(in_pol_labels)
        self.out_pol_labels = list(out_pol_labels)
        assert matrix.shape == (2, 2)
        if self.is_cupy:
            # Build a custom kernel that applies the matrix multiplication.
            # This is much more complicated than the CPU codepath. This is
            # done because older (up to at least GTX 10-series) GPUs lead
            # to errors from cuBlas when trying to use large chunks.
            m = [f"complex<float>({float(x.real)}f, {float(x.imag)}f)" for x in matrix.flat]
            self._kernel = cp.ElementwiseKernel(
                "complex64 x, complex64 y",
                "complex64 p, complex64 q",
                f"""
                    auto x_ = x, y_ = y;  // Copy before overwriting
                    p = {m[0]} * x_ + {m[1]} * y_;
                    q = {m[2]} * x_ + {m[3]} * y_;
                """,
                "convert_polarisation",
            )
        else:
            self._matrix = xr.DataArray(
                matrix.astype(np.complex64),
                dims=("pol", "in_pol"),
                coords={
                    "pol": self.out_pol_labels,
                    "in_pol": self.in_pol_labels,
                },
            )

    async def _transform(self, chunk: xr.DataArray) -> xr.DataArray:
        # This ensures that the ordering matches too. That's important
        # for the cupy path since it sets the output coordinates in
        # order rather than based on the input ordering.
        if chunk.coords["pol"].values.tolist() != self.in_pol_labels:
            raise RuntimeError("Chunk pol coordinates do not match in_pol_labels")
        if self.is_cupy:
            pol0 = chunk.sel({"pol": self.in_pol_labels[0]})
            pol1 = chunk.sel({"pol": self.in_pol_labels[1]})
            # Transform in-place
            self._kernel(pol0.data, pol1.data, pol0.data, pol1.data)
            # Relabel
            return chunk.assign_coords(pol=self.out_pol_labels)
        else:
            out = self._matrix.dot(chunk.rename({"pol": "in_pol"}), dim="in_pol")
            out.attrs = chunk.attrs
            return out
