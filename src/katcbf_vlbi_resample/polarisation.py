# Copyright (c) 2025, National Research Foundation (SARAO)

"""Convert polarised data between bases.

Note that the equations in this module require that the electric field sign
convention is such that phase *increases* with time at a given position in
space. That is valid for MeerKAT for *not* for KAT-7.
"""

import math
import re

import numpy as np

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
    """Parse a single polarisation in the :opt:`--polarisation` command-line option.

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
    """Parse one side of the :opt:`!--polarisation` command-line option.

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
    """Parse the :opt:`!--polarisation` command-line option.

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
