# Copyright (c) 2024, National Research Foundation (SARAO)

"""Encode data to VDIF frames."""

from typing import Any, Iterator

import astropy.units as u
import numpy as np
import xarray as xr
from astropy.time import TimeDelta
from baseband.vdif import VDIFFrame, VDIFFrameSet, VDIFHeader

from .stream import Stream
from .utils import concat_time


class VDIFEncoder:
    """Encode data to VDIF frames.

    The power levels must have already been adjusted such that baseband
    will quantise correctly.
    """

    def __init__(
        self,
        input_data: Stream[xr.DataArray],
        threads: list[dict[str, Any]],
        *,
        bps: int,
        station: str | int,
        samples_per_frame: int,
    ) -> None:
        if input_data.channels is not None:
            raise ValueError("VDIFEncoder currently only supports unchannelised data")
        self._input_it = iter(input_data)
        self._header = VDIFHeader.fromvalues(
            bps=bps,
            complex_data=False,
            nchan=1,
            station=station,
            samples_per_frame=samples_per_frame,
            edv=0,
        )
        self._threads = threads

        self.time_base = input_data.time_base
        self.time_scale = input_data.time_scale
        self.channels = None

    def _frame(self, thread_id: int, header: VDIFHeader, frame_data: xr.DataArray) -> VDIFFrame:
        assert frame_data.dims == ("time",)
        header = header.copy()
        header.update(thread_id=thread_id)
        # The newaxis is to insert the channel dimension (of size 1)
        return VDIFFrame.fromdata(frame_data.to_numpy()[:, np.newaxis], header)

    def _frame_set(self, frame_data: xr.DataArray) -> VDIFFrameSet:
        header = self._header.copy()

        time_offset = frame_data.attrs["time_bias"] * self.time_scale
        # Manually split time_offset (a Fraction) into seconds and fractions
        # of a second, to give a high-accuracy TimeDelta.
        time_offset_secs = int(time_offset)
        time_offset_frac = float(time_offset - time_offset_secs)
        time = self.time_base + TimeDelta(time_offset_secs, time_offset_frac, scale="tai", format="sec")
        frame_rate = float(1 / (header.samples_per_frame * self.time_scale))
        header.set_time(time, frame_rate=frame_rate * u.Hz)

        frames = [self._frame(i, header, frame_data.sel(thread_idx)) for i, thread_idx in enumerate(self._threads)]
        return VDIFFrameSet(frames, header)

    def __iter__(self) -> Iterator[VDIFFrameSet]:
        # TODO: validate that samples_per_frame lines up with time_scale to
        # give an integer number of frames per second. That might require
        # the constructor to have knowledge of time_scale.
        samples_per_frame = self._header.samples_per_frame
        buffer = None
        for in_data in self._input_it:
            if buffer is None:
                buffer = in_data
            else:
                buffer = concat_time([buffer, in_data])
            n_frames = buffer.sizes["time"] // samples_per_frame
            for i in range(n_frames):
                sample_start = i * samples_per_frame
                sample_stop = (i + 1) * samples_per_frame
                frame_data = buffer.isel(time=np.s_[sample_start:sample_stop])
                frame_data.attrs["time_bias"] += sample_start
                yield self._frame_set(frame_data)
            # Cut off the piece that's been processed
            skip = n_frames * samples_per_frame
            buffer = buffer.isel(time=np.s_[skip:])
            buffer.attrs["time_bias"] += skip
