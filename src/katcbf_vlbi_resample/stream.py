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

"""Abstract definition of a sample stream."""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from fractions import Fraction
from typing import Protocol

from astropy.time import Time


class Stream[T](Protocol):
    """Abstract definition of a sample stream.

    See :doc:`design` for details of the fields.
    """

    @property
    def time_base(self) -> Time:  # noqa: D102
        raise NotImplementedError  # pragma: nocover

    @property
    def time_scale(self) -> Fraction:  # noqa: D102
        raise NotImplementedError  # pragma: nocover

    @property
    def channels(self) -> int | None:  # noqa: D102
        raise NotImplementedError  # pragma: nocover

    @property
    def is_cupy(self) -> bool:  # noqa: D102
        raise NotImplementedError  # pragma: nocover

    def __iter__(self) -> Iterator[T]:
        raise NotImplementedError  # pragma: nocover


class ChunkwiseStream[O, I](ABC):
    """Stream where each input chunk becomes one output chunk.

    The chunks need not have the same shape, but if the time scale changes
    then the constructor must override :attr:`time_scale`. By default,
    properties are inherited from the input.

    Subclasses must implement :meth:`_transform`.
    """

    def __init__(self, input_data: Stream[I]) -> None:
        self.time_base = input_data.time_base
        self.time_scale = input_data.time_scale
        self.channels = input_data.channels
        self.is_cupy = input_data.is_cupy
        self._input_it = iter(input_data)

    @abstractmethod
    def _transform(self, chunk: I) -> O:
        """Compute an output chunk from an input chunk."""
        raise NotImplementedError  # pragma: nocover

    def __iter__(self) -> Iterator[O]:
        return self

    def __next__(self) -> O:
        return self._transform(next(self._input_it))
