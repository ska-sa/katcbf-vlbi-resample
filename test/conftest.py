################################################################################
# Copyright (c) 2024, National Research Foundation (SARAO)
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

"""Common fixtures."""

from fractions import Fraction

import cupy as cp
import numpy as np
import pytest
from astropy.time import Time


@pytest.fixture
def time_base() -> Time:
    """Time base for input stream."""
    return Time("2024-07-20T12:00:00.25", scale="utc")


@pytest.fixture
def time_scale() -> Fraction:
    """Time scale for input stream."""
    return Fraction(1, 160000)


@pytest.fixture(params=[pytest.param(np, id="numpy"), pytest.param(cp, id="cupy")])
def xp(request: pytest.FixtureRequest):
    """Module for numpy-like computations.

    Using this fixture allows tests to be performed for both numpy and cupy
    backends.
    """  # noqa: D401
    return request.param
