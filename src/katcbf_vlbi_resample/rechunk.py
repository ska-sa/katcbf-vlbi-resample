# Copyright (c) 2024, National Research Foundation (SARAO)

"""Split and combine chunks to align in time."""

from typing import Final, Iterator

import cupy as cp
import numpy as np
import xarray as xr

from .stream import Stream
from .utils import isel_time


class Rechunk:
    """Split and combine chunks to align in time.

    Note that this requires the incoming stream to be contiguous in time.

    Parameters
    ----------
    input_data
        Incoming data stream.
    samples_per_chunk
        Number of samples per chunk. The `time_bias` of the first sample index
        in each chunk will be a multiple of this plus `remainder`.
    remainder
        See `samples_per_chunk`.
    partial
        If set to true, partial chunks at the start and end may be emitted.
    """

    def __init__(
        self, input_data: Stream[xr.DataArray], samples_per_chunk: int, remainder: int = 0, *, partial: bool = False
    ) -> None:
        self._input_it: Final = iter(input_data)
        self.samples_per_chunk: Final = samples_per_chunk
        self.remainder: Final = remainder
        self.partial: Final = partial

        self.time_base: Final = input_data.time_base
        self.time_scale: Final = input_data.time_scale
        self.is_cupy: Final = input_data.is_cupy
        self.channels: Final = input_data.channels

    def _split_input(self) -> Iterator[xr.DataArray]:
        """Split input chunks on samples_per_chunk boundaries."""
        samples_per_chunk = self.samples_per_chunk
        remainder = self.remainder
        for input_chunk in self._input_it:
            start: int = input_chunk.attrs["time_bias"]
            stop = start + input_chunk.sizes["time"]
            # boundary > start and aligned to samples_per_chunk
            boundary = ((start - remainder) // samples_per_chunk + 1) * samples_per_chunk + remainder
            last = start
            while boundary < stop:
                yield isel_time(input_chunk, np.s_[last - start : boundary - start])
                last = boundary
                boundary += samples_per_chunk
            if last < stop:
                yield isel_time(input_chunk, np.s_[last - start : stop - start])

    def __iter__(self) -> Iterator[xr.DataArray]:
        buffer: xr.DataArray | None = None
        # Copy of buffer.time_bias, for quick access
        buffer_base = -1
        # Range of *valid* samples in buffer, as relative positions in the buffer
        buffer_start = 0
        buffer_stop = 0
        samples_per_chunk = self.samples_per_chunk
        remainder = self.remainder

        def yield_buffer() -> Iterator[xr.DataArray]:
            nonlocal buffer, buffer_base, buffer_start, buffer_stop
            if buffer is None:
                return
            # TODO: would it be safe just to reuse buffer after yielding?
            # Depends on whether downstream consumers finish with it before
            # asking for more.
            if buffer_start == 0 and buffer_stop == samples_per_chunk:
                yield buffer
                buffer = xr.zeros_like(buffer)
            elif self.partial and buffer_start < buffer_stop:
                yield isel_time(buffer, np.s_[buffer_start:buffer_stop])
                buffer = xr.zeros_like(buffer)
            buffer_base += buffer_stop
            buffer_start = 0
            buffer_stop = 0
            # mypy seems to think buffer could be None at this point
            buffer.attrs["time_bias"] = buffer_base  # type: ignore

        xp = cp if self.is_cupy else np
        for input_chunk in self._split_input():
            chunk_start = input_chunk.attrs["time_bias"]
            chunk_size = input_chunk.sizes["time"]
            if chunk_size == 0:
                continue  # Simplifies some corner cases
            boundary = (chunk_start - remainder) // samples_per_chunk * samples_per_chunk + remainder
            if buffer is None:
                shape = list(input_chunk.data.shape)
                # See https://github.com/pydata/xarray/issues/9822
                shape[input_chunk.get_axis_num("time")] = samples_per_chunk  # type: ignore
                buffer = xr.DataArray(
                    xp.zeros(tuple(shape), input_chunk.dtype),
                    dims=input_chunk.dims,
                    coords=input_chunk.coords,
                    attrs=input_chunk.attrs,
                )
                buffer.attrs["time_bias"] = boundary
                buffer_base = boundary
                buffer_start = chunk_start - boundary
                buffer_stop = buffer_start

            if buffer_base + buffer_stop != chunk_start:
                raise ValueError("Input chunks are not contiguous in time")
            assert buffer_stop + chunk_size <= samples_per_chunk

            # Fast path: use chunks as-is if they're already suitable
            if boundary == chunk_start and chunk_size == samples_per_chunk:
                yield input_chunk
                buffer_base += samples_per_chunk
                buffer.attrs["time_bias"] = buffer_base
            else:
                buffer[dict(time=np.s_[buffer_stop : buffer_stop + chunk_size])] = input_chunk
                buffer_stop += chunk_size
                if buffer_stop == samples_per_chunk:
                    # We've reached the end of a chunk
                    yield from yield_buffer()

        yield from yield_buffer()  # Deal with any trailing piece
