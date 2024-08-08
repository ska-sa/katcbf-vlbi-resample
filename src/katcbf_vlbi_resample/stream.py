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
