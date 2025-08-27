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

import asyncio
from collections import deque
from collections.abc import AsyncIterator

import cupy as cp
import cupyx
import packaging.version
import xarray as xr

from .stream import ChunkwiseStream, Stream
from .utils import as_cupy, stream_future

# We don't list cupy in project requirements because there are multiple
# packages that could provide it (e.g. cupy-cudaNNx for binary wheels),
# so we check the dependencies at import time.
_cupy_version = packaging.version.Version(cp.__version__)
if _cupy_version < packaging.version.Version("13.3"):
    raise ImportError("cupy >= 13.4 is required", name="cupy")
if _cupy_version == packaging.version.Version("13.5.1"):
    raise ImportError("cupy 13.5.1 is not supported due to a bug", name="cupy")


class AsCupy(ChunkwiseStream[xr.DataArray, xr.DataArray]):
    """Transfer a stream from numpy to cupy.

    The transfer is enqueued to the current CUDA stream, but this does
    not block on the transfer completing.
    """

    def __init__(self, input_data: Stream[xr.DataArray]) -> None:
        super().__init__(input_data)
        self.is_cupy = True

    async def _transform(self, chunk: xr.DataArray) -> xr.DataArray:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, as_cupy, chunk)


class AsNumpy:
    """Transfer a stream from cupy to numpy."""

    def __init__(self, input_data: Stream[xr.DataArray], queue_depth: int = 1) -> None:
        self.time_base = input_data.time_base
        self.time_scale = input_data.time_scale
        self.channels = input_data.channels
        self.is_cupy = False
        self._input_it = aiter(input_data)
        self._queue: deque[asyncio.Future[xr.DataArray]] = deque()
        self._queue_depth = queue_depth

    async def _flush(self, max_size: int) -> AsyncIterator[xr.DataArray]:
        while len(self._queue) > max_size:
            yield (await self._queue.popleft())

    async def __aiter__(self) -> AsyncIterator[xr.DataArray]:
        async for buffer in self._input_it:
            out = cupyx.empty_like_pinned(buffer.data)
            buffer = buffer.copy(data=cp.asnumpy(buffer.data, out=out, blocking=False))
            self._queue.append(stream_future(buffer))
            async for chunk in self._flush(self._queue_depth):
                yield chunk
        async for chunk in self._flush(0):
            yield chunk
