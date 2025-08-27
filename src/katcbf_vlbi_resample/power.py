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

"""Normalise power level prior to quantisation."""

from abc import abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import Literal

import cupy as cp
import cupyx
import numpy as np
import xarray as xr

from .stream import ChunkwiseStream, Stream
from .utils import as_cupy, wait_event


def _rms(array: np.ndarray | cp.ndarray) -> np.ndarray | cp.ndarray:
    if isinstance(array, cp.ndarray):
        kernel = cp.ReductionKernel(
            "T x, T scale",  # inputs
            "T y",  # outputs
            "x * x",  # map
            "a + b",  # reduce
            "y = sqrt(a / scale)",  # post-reduction map
            "0",  # identity
            "rms",  # kernel name
        )
        return kernel(array, np.float32(array.shape[-1]), axis=-1)
    else:
        return np.sqrt(np.square(array).mean(axis=-1))


@dataclass(slots=True)
class _RmsHistoryEntry:
    start: int  # First sample index
    length: int  # Number of samples
    rms: xr.DataArray  # Numpy-backed RMS
    event: cp.cuda.Event  # Event to wait for rms to be valid


class MeasurePower(ChunkwiseStream[xr.Dataset, xr.DataArray]):
    """Measure mean power in each chunk.

    It may be beneficial to use :class:`.Rechunk` prior to this filter to
    control the chunk size.

    The output stream contains Datasets rather than DataArrays. The Dataset
    has members `data` (containing the original data) and `rms` (containing
    the RMS voltages without a time axis).
    """

    async def _transform(self, chunk: xr.DataArray) -> xr.Dataset:
        assert chunk.dtype.kind == "f", "only real floating-point data is supported"
        rms = xr.apply_ufunc(_rms, chunk, input_core_dims=[["time"]], output_dtypes=[chunk.dtype])
        rms.name = "rms"
        return xr.Dataset({"data": chunk, "rms": rms})


class RecordPower(ChunkwiseStream[xr.Dataset, xr.Dataset]):
    """Record history of power levels.

    This stream should be inserted immediately after :class:`MeasurePower`.
    It passes the chunks through unchanged, but invokes a callback function
    to save power levels.
    """

    _MAX_HISTORY = 64  # Maximum depth for rms_history deque

    def __init__(self, input_data: Stream[xr.Dataset]) -> None:
        super().__init__(input_data)
        self._rms_history: deque[_RmsHistoryEntry] = deque()

    @abstractmethod
    def record_rms(self, start: int, length: int, rms: xr.DataArray) -> None:
        """Record the RMS values.

        The base class does nothing. Subclasses may override it to take action.

        The `rms` array is guaranteed to be backed by a numpy array rather than
        GPU memory. To make this efficient when using cupy (and not block the
        pipeline to transfer data to the host), there may be some delay between
        the chunk being iterated and this callback.

        Parameters
        ----------
        start
            Sample index of the first sample in the chunk
        length
            Number of samples in the chunk
        rms
            RMS values from the chunk (with non-time dimensions preserved).
        """
        pass  # pragma: nocover

    async def _transform(self, chunk: xr.Dataset) -> xr.Dataset:
        # Duplicate since the downstream owns the chunk once it's yielded
        rms = chunk["rms"].copy()
        data = chunk["data"]
        if isinstance(rms.data, cp.ndarray):
            rms_out = cupyx.empty_like_pinned(rms.data)
            rms_np = rms.copy(data=cp.asnumpy(rms.data, out=rms_out, blocking=False))
            event = cp.cuda.Event(disable_timing=True)
            event.record()
            entry = _RmsHistoryEntry(data.attrs["time_bias"], data.sizes["time"], rms_np, event)
            self._rms_history.append(entry)
            if len(self._rms_history) > self._MAX_HISTORY:
                await wait_event(self._rms_history[0].event)
            while self._rms_history and self._rms_history[0].event.done:
                entry = self._rms_history.popleft()
                self.record_rms(entry.start, entry.length, entry.rms)
        else:
            self.record_rms(data.attrs["time_bias"], data.sizes["time"], rms)

        return chunk

    async def __anext__(self) -> xr.Dataset:
        try:
            return await super().__anext__()
        except StopAsyncIteration:
            # Flush out _rms_history
            while self._rms_history:
                entry = self._rms_history.popleft()
                await wait_event(entry.event)
                self.record_rms(entry.start, entry.length, entry.rms)
            raise


class NormalisePower(ChunkwiseStream[xr.DataArray, xr.Dataset]):
    """Normalise power level.

    The power level is adjusted so that the standard deviation within each
    chunk is `scale`. This is done independently for each time series. It
    may be beneficial to use :class:`.Rechunk` prior to this filter to
    control the chunk size.

    Subclasses may override :meth:`record_rms` to store the RMS values to some
    form of storage.

    Parameters
    ----------
    input_data
        Input data stream. The input chunks must be Datasets with `data` and
        `rms` members, as yielded by :class:`MeasurePower`. If `power` is float,
        the `rms` member is not required.
    scale
        Target standard deviation
    power
        Method used to determine the power for the normalisation factor. This may be one of

        ``auto``
            Use the values from the `rms` member of the input chunks.
        ``first``
            Use the values from the `rms` member of the first input chunk, for all chunks.
        DataArray
            Constant values to use as the power estimates. This must have the same non-time
            dimensions as the data.
    """

    def __init__(
        self, input_data: Stream[xr.Dataset], scale: float, power: Literal["auto", "first"] | xr.DataArray = "auto"
    ) -> None:
        super().__init__(input_data)
        self.scale = scale
        self.power = power
        self._mul: xr.DataArray | None = None  # Factor to multiply by
        if not isinstance(power, xr.DataArray) and power not in {"auto", "first"}:
            raise ValueError("power must be 'auto', 'first' or a DataArray")

    async def _transform(self, chunk: xr.Dataset) -> xr.DataArray:
        if self._mul is not None:
            mul = self._mul
        else:
            if isinstance(self.power, xr.DataArray):
                # mypy believes np.sqrt returns an ndarray, but xarray overloads it
                mul = self.scale / np.sqrt(self.power)  # type: ignore
                mul = mul.reindex_like(chunk["data"], copy=False)
                if self.is_cupy:
                    mul = as_cupy(mul, blocking=False)
                else:
                    mul = mul.as_numpy()
                self._mul = mul
            else:
                mul = self.scale / chunk["rms"]
                if self.power == "first":
                    self._mul = mul  # Reuse it for all future chunks
        data = chunk["data"]
        data *= mul  # TODO: is it safe to modify in place?
        return data
