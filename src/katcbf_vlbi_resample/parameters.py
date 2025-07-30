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

"""Dataclasses to encapsulate configuration."""

from dataclasses import dataclass


@dataclass(frozen=True)
class StreamParameters:
    """Parameters of a time-domain stream of samples."""

    bandwidth: float  # Hz
    center_freq: float  # Hz


@dataclass(frozen=True)
class ResampleParameters:
    """Parameters controlling the resampling process."""

    fir_taps: int
    hilbert_taps: int
    passband: float
