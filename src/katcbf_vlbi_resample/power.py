# Copyright (c) 2024, National Research Foundation (SARAO)

"""Normalise power level prior to quantisation."""

import numpy as np
import xarray as xr

from .stream import ChunkwiseStream, Stream


class NormalisePower(ChunkwiseStream[xr.DataArray, xr.DataArray]):
    """Normalise power level.

    The power level is adjusted so that the standard deviation is `scale`.
    This is done independently for each time series.

    TODO: this does not yet use a moving window.
    """

    def __init__(self, input_data: Stream[xr.DataArray], scale: float) -> None:
        super().__init__(input_data)
        self.scale = scale

    def _transform(self, chunk: xr.DataArray) -> xr.DataArray:
        assert chunk.dtype.kind == "f", "only real floating-point data is supported"
        # mypy doesn't recognise that np.square returns an xarray object in this case.
        rms = np.sqrt(np.square(chunk).mean(dim="time"))  # type: ignore
        chunk *= self.scale / rms  # TODO: is it safe to modify in place?
        return chunk
