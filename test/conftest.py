# Copyright (c) 2024, National Research Foundation (SARAO)

"""Common fixtures."""

from fractions import Fraction

import cupy as cp
import numpy as np
import pytest
from astropy.time import Time


@pytest.fixture
def time_base() -> Time:
    """Time base for input stream."""
    return Time("2024-07-20T12:00:00", scale="utc")


@pytest.fixture
def time_scale() -> Fraction:
    """Time scale for input stream."""
    return Fraction(1, 1234)


@pytest.fixture(params=[pytest.param(np, id="numpy"), pytest.param(cp, id="cupy")])
def xp(request: pytest.FixtureRequest):
    """Module for numpy-like computations.

    Using this fixture allows tests to be performed for both numpy and cupy
    backends.
    """  # noqa: D401
    return request.param
