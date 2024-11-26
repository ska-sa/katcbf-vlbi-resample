# Copyright (c) 2024, National Research Foundation (SARAO)

"""Normalise power level prior to quantisation."""

from collections import deque
from dataclasses import dataclass

import cupy as cp
import cupyx
import numpy as np
import xarray as xr

from .stream import ChunkwiseStream, Stream


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


class NormalisePower(ChunkwiseStream[xr.DataArray, xr.DataArray]):
    """Normalise power level.

    The power level is adjusted so that the standard deviation within each
    chunk is `scale`. This is done independently for each time series. It
    may be beneficial to use :class:`.Rechunk` prior to this filter to
    control the chunk size.

    Subclasses may override :meth:`record_rms` to store the RMS values to some
    form of storage.
    """

    _MAX_HISTORY = 64  # Maximum depth for rms_history deque

    def __init__(self, input_data: Stream[xr.DataArray], scale: float) -> None:
        super().__init__(input_data)
        self.scale = scale
        self._rms_history: deque[_RmsHistoryEntry] = deque()

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

    def _transform(self, chunk: xr.DataArray) -> xr.DataArray:
        assert chunk.dtype.kind == "f", "only real floating-point data is supported"
        rms = xr.apply_ufunc(_rms, chunk, input_core_dims=[["time"]], output_dtypes=[chunk.dtype])
        chunk *= self.scale / rms  # TODO: is it safe to modify in place?

        if isinstance(rms.data, cp.ndarray):
            rms_out = cupyx.empty_like_pinned(rms.data)
            rms_np = rms.copy(data=cp.asnumpy(rms.data, out=rms_out, blocking=False))
            event = cp.cuda.Event(disable_timing=True)
            event.record()
            entry = _RmsHistoryEntry(chunk.attrs["time_bias"], chunk.sizes["time"], rms_np, event)
            self._rms_history.append(entry)
            if len(self._rms_history) > self._MAX_HISTORY:
                self._rms_history[0].event.synchronize()
            while self._rms_history and self._rms_history[0].event.done:
                entry = self._rms_history.popleft()
                self.record_rms(entry.start, entry.length, entry.rms)
        else:
            self.record_rms(chunk.attrs["time_bias"], chunk.sizes["time"], rms)

        return chunk

    def __next__(self) -> xr.DataArray:
        try:
            return super().__next__()
        except StopIteration:
            # Flush out _rms_history
            while self._rms_history:
                entry = self._rms_history.popleft()
                entry.event.synchronize()
                self.record_rms(entry.start, entry.length, entry.rms)
            raise
