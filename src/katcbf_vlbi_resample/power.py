# Copyright (c) 2024, National Research Foundation (SARAO)

"""Normalise power level prior to quantisation."""

from collections.abc import Iterable
from typing import Self

import numpy as np
import xarray as xr


class NormalisePower:
    """Normalise power level.

    The power level is adjusted so that the standard deviation is `scale`.
    This is done independently for each time series.

    TODO: this does not yet use a moving window.
    """

    def __init__(self, input_data: Iterable[xr.DataArray], scale: float) -> None:
        self._input_it = iter(input_data)
        self.scale = scale

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> xr.DataArray:
        chunk = next(self._input_it)
        # mypy doesn't recognise that np.square returns an xarray object in this case.
        rms = np.sqrt(np.square(chunk).mean(dim="time"))  # type: ignore
        chunk *= self.scale / rms
        return chunk
