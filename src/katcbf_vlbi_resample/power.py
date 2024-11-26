# Copyright (c) 2024, National Research Foundation (SARAO)

"""Normalise power level prior to quantisation."""

import cupy as cp
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


class NormalisePower(ChunkwiseStream[xr.DataArray, xr.DataArray]):
    """Normalise power level.

    The power level is adjusted so that the standard deviation within each
    chunk is `scale`. This is done independently for each time series. It
    may be beneficial to use :class:`.Rechunk` prior to this filter to
    control the chunk size.
    """

    def __init__(self, input_data: Stream[xr.DataArray], scale: float) -> None:
        super().__init__(input_data)
        self.scale = scale

    def _transform(self, chunk: xr.DataArray) -> xr.DataArray:
        assert chunk.dtype.kind == "f", "only real floating-point data is supported"
        rms = xr.apply_ufunc(_rms, chunk, input_core_dims=[["time"]], output_dtypes=[chunk.dtype])
        chunk *= self.scale / rms  # TODO: is it safe to modify in place?
        return chunk
