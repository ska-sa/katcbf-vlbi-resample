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

"""Convert streams to and from cupy."""

from collections import deque
from collections.abc import AsyncIterator

import cupy as cp
import cupyx
import xarray as xr

from .stream import ChunkwiseStream, Stream
from .utils import as_cupy


class AsCupy(ChunkwiseStream[xr.DataArray, xr.DataArray]):
    """Transfer a stream from numpy to cupy."""

    def __init__(self, input_data: Stream[xr.DataArray]) -> None:
        super().__init__(input_data)
        self.is_cupy = True

    async def _transform(self, chunk: xr.DataArray) -> xr.DataArray:
        # TODO: make this properly asynchronous
        return as_cupy(chunk, blocking=True)


class AsNumpy:
    """Transfer a stream from cupy to numpy."""

    def __init__(self, input_data: Stream[xr.DataArray], queue_depth: int = 1) -> None:
        self.time_base = input_data.time_base
        self.time_scale = input_data.time_scale
        self.channels = input_data.channels
        self.is_cupy = False
        self._input_it = aiter(input_data)
        self._queue: deque[tuple[xr.DataArray, cp.cuda.Event]] = deque()
        self._queue_depth = queue_depth

    async def _flush(self, max_size: int) -> AsyncIterator[xr.DataArray]:
        while len(self._queue) > max_size:
            buffer, event = self._queue.popleft()
            event.synchronize()  # TODO: make properly asynchronous
            yield buffer

    async def __aiter__(self) -> AsyncIterator[xr.DataArray]:
        async for buffer in self._input_it:
            out = cupyx.empty_like_pinned(buffer.data)
            buffer = buffer.copy(data=cp.asnumpy(buffer.data, out=out, blocking=False))
            event = cp.cuda.Event(disable_timing=True)
            event.record()
            self._queue.append((buffer, event))
            async for chunk in self._flush(self._queue_depth):
                yield chunk
        async for chunk in self._flush(0):
            yield chunk
