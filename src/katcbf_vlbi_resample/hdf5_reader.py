# Copyright (c) 2024, National Research Foundation (SARAO)

"""Load data from MeerKAT beamformer HDF5 files."""

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Self, TypeVar

import h5py
import numpy as np
import xarray as xr
from astropy.time import Time, TimeDelta

_T = TypeVar("_T")


def _single_value(name: str, values: Sequence[_T]) -> _T:
    """Check that all values in `values` are the same, and return that value.

    Raises
    ------
    ValueError
        if the values are not all the same
    """
    if not values:
        raise ValueError(f"No values found for {name}")
    for value in values:
        if value != values[0]:
            raise ValueError(f"Inconsistent values for {name} ({value} != {values[0]})")
    return value


@dataclass
class _HDF5Input:
    name: str
    file: h5py.File
    bf_raw: h5py.Dataset
    channels: int
    spectra: int  # Number of spectra in the file
    first_ts_adc: int  # First timestamp, in ADC samples
    step_ts_adc: int  # Step between spectra, in ADC samples
    last_ts_adc: int  # Past-the-end timestamp
    offset: int  # First spectrum to load

    @classmethod
    def from_file(cls, name: str, file: h5py.File) -> Self:
        """Load from a file.

        The `offset` parameter is set to 0 for population later.
        """
        bf_raw = file["Data"]["bf_raw"]
        timestamps = file["Data"]["timestamps"]
        spectra = int(bf_raw.shape[1])
        first_ts_adc = int(timestamps[0])
        # katsdpbfingest always writes regularly-spaced timestamps
        step_ts_adc = int(timestamps[1]) - first_ts_adc
        if first_ts_adc % step_ts_adc != 0:
            raise ValueError(f"File {name} has misaligned timestamps")

        return cls(
            name=name,
            file=file,
            bf_raw=bf_raw,
            channels=int(bf_raw.shape[0]),
            spectra=spectra,
            first_ts_adc=first_ts_adc,
            step_ts_adc=step_ts_adc,
            last_ts_adc=first_ts_adc + step_ts_adc * spectra,
            offset=0,
        )


class HDF5Reader:
    """Load data from a MeerKAT beamformer HDF5 file."""

    def __init__(
        self,
        files: Mapping[str, h5py.File],
        adc_sample_rate: float,  # Hz
        sync_time: Time,
        start_time: Time | None = None,
        duration: TimeDelta | None = None,
    ) -> None:
        self._inputs = [_HDF5Input.from_file(name, file) for name, file in files.items()]
        if not self._inputs:
            raise ValueError("At least one input must be defined")

        step_ts_adc = _single_value("timestamp step", [f.step_ts_adc for f in self._inputs])
        self.time_base = sync_time
        self.time_scale = Fraction(step_ts_adc) / Fraction(adc_sample_rate)
        self.channels = _single_value("channels", [f.channels for f in self._inputs])
        self.is_cupy = False

        first_ts_adc = max(f.first_ts_adc for f in self._inputs)
        last_ts_adc = min(f.last_ts_adc for f in self._inputs)
        if last_ts_adc <= first_ts_adc:
            raise ValueError("The input files do not overlap in time")

        if start_time is not None:
            start_ts_adc = round((start_time - sync_time).tai.sec * adc_sample_rate)
            start_ts_adc = start_ts_adc // step_ts_adc * step_ts_adc
        else:
            start_ts_adc = first_ts_adc
        if duration is not None:
            stop_ts_adc = start_ts_adc + round(duration.tai.sec * adc_sample_rate)
            # Round outwards to ensure we get all the desired samples
            stop_ts_adc = (stop_ts_adc - step_ts_adc - 1) // step_ts_adc * step_ts_adc
        else:
            stop_ts_adc = last_ts_adc
        if stop_ts_adc <= first_ts_adc or start_ts_adc >= last_ts_adc:
            raise ValueError("No overlap between requested and available time ranges")

        self._start_ts_adc = max(start_ts_adc, first_ts_adc)
        self._stop_ts_adc = min(stop_ts_adc, last_ts_adc)
        self._step_ts_adc = step_ts_adc
        for f in self._inputs:
            f.offset = -f.first_ts_adc // step_ts_adc

    def __iter__(self) -> Iterator[xr.DataArray]:
        chunk_spectra = self._inputs[0].bf_raw.chunks[1]
        chunk_adc = self._step_ts_adc * chunk_spectra
        for ts0 in range(self._start_ts_adc, self._stop_ts_adc, chunk_adc):
            ts1 = min(ts0 + chunk_adc, self._stop_ts_adc)
            spectrum0 = ts0 // self._step_ts_adc
            spectrum1 = ts1 // self._step_ts_adc
            parts = []
            for f in self._inputs:
                parts.append(f.bf_raw[:, f.offset + spectrum0 : f.offset + spectrum1, :])
            # Combine, then convert to complex. TODO: nan out missing data
            data = np.stack(parts).astype(np.float64).view(np.complex128)[..., 0]
            yield xr.DataArray(
                data,
                dims=("pol", "channel", "time"),
                coords={"pol": [f.name for f in self._inputs]},
                attrs={"time_bias": ts0 // self._step_ts_adc},
            )
        for f in self._inputs:
            f.file.close()
