# Copyright (c) 2024, National Research Foundation (SARAO)

"""Convert streams to and from cupy."""

from collections import deque
from collections.abc import Iterator
from typing import Self

import cupy as cp
import cupyx
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

    def __init__(self, input_data: Stream[xr.DataArray], queue_depth: int = 1) -> None:
        self.time_base = input_data.time_base
        self.time_scale = input_data.time_scale
        self.channels = input_data.channels
        self.is_cupy = False
        self._input_it = iter(input_data)
        self._queue: deque[tuple[xr.DataArray, cp.cuda.Event]] = deque()
        self._queue_depth = queue_depth

    def _flush(self, max_size: int) -> Iterator[xr.DataArray]:
        while len(self._queue) > max_size:
            buffer, event = self._queue.popleft()
            event.synchronize()
            yield buffer

    def __iter__(self) -> Iterator[xr.DataArray]:
        for buffer in self._input_it:
            out = cupyx.empty_like_pinned(buffer.data)
            buffer = xr.DataArray(
                cp.asnumpy(buffer.data, out=out, blocking=False),
                dims=buffer.dims,
                coords=buffer.coords,
                name=buffer.name,
                attrs=buffer.attrs,
            )
            event = cp.cuda.Event(disable_timing=True)
            event.record()
            self._queue.append((buffer, event))
            yield from self._flush(self._queue_depth)
        yield from self._flush(0)
