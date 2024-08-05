# Copyright (c) 2024, National Research Foundation (SARAO)

"""Convert streams to and from cupy."""

from typing import Self

import xarray as xr

from .stream import Stream
from .utils import as_cupy


class AsCupy:
    """Transfer a stream from numpy to cupy."""

    def __init__(self, input_data: Stream[xr.DataArray]) -> None:
        self.time_base = input_data.time_base
        self.time_scale = input_data.time_scale
        self.channels = input_data.channels
        self.is_cupy = True
        self._input_it = iter(input_data)

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> xr.DataArray:
        return as_cupy(next(self._input_it))


class AsNumpy:
    """Transfer a stream from cupy to numpy."""

    def __init__(self, input_data: Stream[xr.DataArray]) -> None:
        self.time_base = input_data.time_base
        self.time_scale = input_data.time_scale
        self.channels = input_data.channels
        self.is_cupy = False
        self._input_it = iter(input_data)

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> xr.DataArray:
        return next(self._input_it).as_numpy()
