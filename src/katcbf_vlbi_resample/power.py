# Copyright (c) 2024, National Research Foundation (SARAO)

"""Normalise power level prior to quantisation."""

from typing import Self

import numpy as np
import xarray as xr

from .stream import Stream


class NormalisePower:
    """Normalise power level.

    The power level is adjusted so that the standard deviation is `scale`.
    This is done independently for each time series.

    TODO: this does not yet use a moving window.
    """

    def __init__(self, input_data: Stream[xr.DataArray], scale: float) -> None:
        self._input_it = iter(input_data)
        self.scale = scale
        self.time_base = input_data.time_base
        self.time_scale = input_data.time_scale
        self.channels = input_data.channels

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> xr.DataArray:
        chunk = next(self._input_it)
        assert chunk.dtype.kind == "f", "only real floating-point data is supported"
        # mypy doesn't recognise that np.square returns an xarray object in this case.
        rms = np.sqrt(np.square(chunk).mean(dim="time"))  # type: ignore
        chunk *= self.scale / rms
        return chunk
