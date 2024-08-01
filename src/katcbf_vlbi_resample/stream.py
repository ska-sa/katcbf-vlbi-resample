# Copyright (c) 2024, National Research Foundation (SARAO)

"""Abstract definition of a sample stream."""

from collections.abc import Iterator
from fractions import Fraction
from typing import Protocol, TypeVar

from astropy.time import Time

_T_co = TypeVar("_T_co", covariant=True)


class Stream(Protocol[_T_co]):
    """Abstract definition of a sample stream.

    See :doc:`design` for details of the fields.
    """

    time_base: Time
    time_scale: Fraction
    channels: int | None  # None means no channel axis
    is_cupy: bool

    def __iter__(self) -> Iterator[_T_co]:
        pass  # pragma: nocover
