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
from typing import Generic, Protocol, TypeVar

from astropy.time import Time

_T_co = TypeVar("_T_co", covariant=True)
_T_contra = TypeVar("_T_contra", contravariant=True)


class Stream(Protocol[_T_co]):
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

    def __iter__(self) -> Iterator[_T_co]:
        raise NotImplementedError  # pragma: nocover


class ChunkwiseStream(ABC, Generic[_T_co, _T_contra]):
    """Stream where each input chunk becomes one output chunk.

    The chunks need not have the same shape, but if the time scale changes
    then the constructor must override :attr:`time_scale`. By default,
    properties are inherited from the input.

    Subclasses must implement :meth:`_transform`.
    """

    def __init__(self, input_data: Stream[_T_contra]) -> None:
        self.time_base = input_data.time_base
        self.time_scale = input_data.time_scale
        self.channels = input_data.channels
        self.is_cupy = input_data.is_cupy
        self._input_it = iter(input_data)

    @abstractmethod
    def _transform(self, chunk: _T_contra) -> _T_co:
        """Compute an output chunk from an input chunk."""
        raise NotImplementedError  # pragma: nocover

    def __iter__(self) -> Iterator[_T_co]:
        return self

    def __next__(self) -> _T_co:
        return self._transform(next(self._input_it))
