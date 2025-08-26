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

"""Load data from MeerKAT beamformer HDF5 files."""

import asyncio
from collections import deque
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Self

import cupy as cp
import cupyx
import h5py
import numpy as np
import xarray as xr
from astropy.time import Time, TimeDelta


def _single_value[T](name: str, values: Sequence[T]) -> T:
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


class _PinnedBuffer:
    """Transfer pinned in CUDA pinned memory."""

    def __init__(self, shape: tuple[int, ...], dtype: np.dtype) -> None:
        self._data = cupyx.empty_pinned(shape, dtype)
        self._event: cp.cuda.Event | None = None

    async def get(self) -> np.ndarray:
        """Get the array, waiting for any recorded event."""
        if self._event is not None:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._event.synchronize)
        return self._data

    def put(self, event: cp.cuda.Event) -> None:
        """Record that the array is in use for an asynchronous transfer."""
        self._event = event


class HDF5Reader:
    """Load data from a MeerKAT beamformer HDF5 file."""

    def __init__(
        self,
        files: Mapping[str, h5py.File],
        adc_sample_rate: float,  # Hz
        sync_time: Time,
        start_time: Time | None = None,
        duration: TimeDelta | None = None,
        is_cupy: bool = False,
    ) -> None:
        self._inputs = [_HDF5Input.from_file(name, file) for name, file in files.items()]
        if not self._inputs:
            raise ValueError("At least one input must be defined")

        step_ts_adc = _single_value("timestamp step", [f.step_ts_adc for f in self._inputs])
        self.time_base = sync_time
        self.time_scale = Fraction(step_ts_adc) / Fraction(adc_sample_rate)
        self.channels = _single_value("channels", [f.channels for f in self._inputs])
        self.is_cupy = is_cupy

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

    @property
    def start_spectrum(self) -> int:
        """Index of first spectrum that will be returned.

        The reference point is the `time_base`. This is thus the first
        `time_bias` that will appear.
        """
        return self._start_ts_adc // self._step_ts_adc

    @property
    def stop_spectrum(self) -> int:
        """Index past the last spectrum that will be returned.

        See :attr:`start_spectrum`.
        """
        return self._stop_ts_adc // self._step_ts_adc

    async def __aiter__(self) -> AsyncIterator[xr.DataArray]:
        chunk_spectra = self._inputs[0].bf_raw.chunks[1]
        chunk_adc = self._step_ts_adc * chunk_spectra
        transfer_shape = (len(self._inputs), self.channels, chunk_spectra, 2)
        dtype = self._inputs[0].bf_raw.dtype
        if self.is_cupy:
            transfer_bufs = deque(_PinnedBuffer(transfer_shape, dtype) for _ in range(3))
        for ts0 in range(self._start_ts_adc, self._stop_ts_adc, chunk_adc):
            ts1 = min(ts0 + chunk_adc, self._stop_ts_adc)
            spectrum0 = ts0 // self._step_ts_adc
            spectrum1 = ts1 // self._step_ts_adc
            if self.is_cupy:
                xp = cp
                transfer_buf = transfer_bufs.popleft()
                store = await transfer_buf.get()
            else:
                xp = np
                store = np.empty(transfer_shape, dtype)
            for i, f in enumerate(self._inputs):
                f.bf_raw.read_direct(
                    store,
                    np.s_[:, f.offset + spectrum0 : f.offset + spectrum1, :],
                    np.s_[i, :, : spectrum1 - spectrum0, :],
                )
            # Trim the array if this is the last chunk and it's short. We
            # don't do this earlier because read_direct wants contiguous memory.
            device_store = xp.asarray(store[:, :, : spectrum1 - spectrum0, :])
            if self.is_cupy:
                event = cp.cuda.Event(disable_timing=True)
                event.record()
                transfer_buf.put(event)
                transfer_bufs.append(transfer_buf)
            # Convert Gaussian integers to complex. TODO: nan out missing data
            # Also transpose the time and channel axes. That's not actually
            # required, but it ends up happening eventually, and it's cheaper
            # to do it early (before converting int8 -> float32).
            yield xr.DataArray(
                xp.require(device_store.transpose(0, 2, 1, 3), np.float32, "C").view(np.complex64)[..., 0],
                dims=("pol", "time", "channel"),
                coords={"pol": [f.name for f in self._inputs]},
                attrs={"time_bias": ts0 // self._step_ts_adc},
            )
        for f in self._inputs:
            f.file.close()
