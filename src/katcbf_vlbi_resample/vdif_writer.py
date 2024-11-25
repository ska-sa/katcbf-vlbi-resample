# Copyright (c) 2024, National Research Foundation (SARAO)

"""Encode data to VDIF frames."""

from typing import Any, Final, Iterator

import astropy.units as u
import cupy as cp
import numpy as np
import xarray as xr
from astropy.time import TimeDelta
from baseband.base.encoding import TWO_BIT_1_SIGMA
from baseband.vdif import VDIFFrame, VDIFFrameSet, VDIFHeader, VDIFPayload

from .stream import Stream
from .utils import concat_time


@cp.fuse
def _encode_2bit_prep(values):
    xp = cp.get_array_module(values)
    return xp.clip(values * (1 / TWO_BIT_1_SIGMA) + 2, 0, 3).astype(np.uint8)


def _encode_2bit(values):
    """Encode values using 2 bits per value, packing the result into bytes.

    This is equivalent to :func:`baseband.vdif.encode_2bit`, but supports
    cupy. Note that values very close to the threshold might round
    differently.
    """
    xp = cp.get_array_module(values)
    values = _encode_2bit_prep(values)
    values = values.reshape(values.shape[:-1] + (-1, 4))
    values <<= xp.arange(0, 8, 2, dtype=np.uint8)
    # cupy doesn't currently support np.bitwise_or.reduce
    return values[..., 0] | values[..., 1] | values[..., 2] | values[..., 3]


def _encode_2bit_words(values):
    return _encode_2bit(values).view("<u4")


class VDIFEncode2Bit:
    """Quantise and pack values to 2-bit samples.

    The power levels must have already been adjusted such that :mod:`baseband`
    will quantise correctly.

    The output array is in units of VDIF words (32-bit little-endian) rather
    than samples, and `time_bias` is also in units of words. Only real-valued
    unchannelised data is supported.

    The input need not have any particular alignment. The output is aligned
    to the VDIF frame length.
    """

    SAMPLES_PER_WORD: Final = 16

    def __init__(self, input_data: Stream[xr.DataArray], samples_per_frame: int) -> None:
        if input_data.channels is not None:
            raise ValueError("VDIFEncoder2Bit currently only supports unchannelised data")
        # VDIF requires an even number of words per frame
        if samples_per_frame % (2 * self.SAMPLES_PER_WORD) != 0:
            raise ValueError(f"samples_per_frame must be a multiple of {2 * self.SAMPLES_PER_WORD}")
        if (samples_per_frame * input_data.time_scale).numerator != 1:
            raise ValueError("samples_per_frame does not yield an integer frame rate")

        self._input_it = iter(input_data)

        # The first sample of each frame must satisfy
        # index % samples_per_frame = phase, where we determine phase here
        # so that the timestamp error is <=0.5 samples.

        # Get fractions of a second until the next second boundary.
        frac = -input_data.time_base.utc.unix % 1
        frac_samples = frac / input_data.time_scale
        self._phase = round(frac_samples) % samples_per_frame

        self.samples_per_frame = samples_per_frame
        self.time_base = input_data.time_base
        self.time_scale = input_data.time_scale * self.SAMPLES_PER_WORD
        self.channels = None
        self.is_cupy = input_data.is_cupy

    def __iter__(self) -> Iterator[VDIFFrameSet]:
        samples_per_frame = self.samples_per_frame
        buffer = None
        for in_data in self._input_it:
            if buffer is None:
                buffer = in_data
            else:
                buffer = concat_time([buffer, in_data])
            # Discard leading partial frame.
            time_bias: int = buffer.attrs["time_bias"]
            if time_bias % samples_per_frame != self._phase:
                trim = (self._phase - time_bias) % samples_per_frame
                if trim >= buffer.sizes["time"]:
                    continue  # Don't have enough data to reach the frame boundary
                buffer = buffer.isel(time=np.s_[trim:])
                buffer.attrs["time_bias"] += trim

            n_frames = buffer.sizes["time"] // samples_per_frame
            encoded = xr.apply_ufunc(
                _encode_2bit_words,
                buffer.isel(time=np.s_[: n_frames * samples_per_frame]),
                input_core_dims=[["time"]],
                output_core_dims=[("time",)],
                exclude_dims={"time"},
                keep_attrs=True,
            )
            assert encoded.attrs["time_bias"] % self.SAMPLES_PER_WORD == 0
            encoded.attrs["time_bias"] //= self.SAMPLES_PER_WORD
            yield encoded
            # Cut off the piece that's been processed
            skip = n_frames * samples_per_frame
            buffer = buffer.isel(time=np.s_[skip:])
            buffer.attrs["time_bias"] += skip


class VDIFFormatter:
    """Encode data to VDIF frames.

    The data must have already been quantised to words using
    :class:`VDIFEncode2Bit` or similar.

    Multiple VDIF threads are supported, provided that they are uniform. The
    `threads` argument must contain one element per desired thread, in the
    order that the thread IDs are to be assigned. The element is a dictionary
    that is passed to :meth:`xarray.DataArray.sel` to select the desired
    subarray from the input chunks.
    """

    def __init__(
        self,
        input_data: Stream[xr.DataArray],
        threads: list[dict[str, Any]],
        *,
        station: str | int,
        samples_per_frame: int,
    ) -> None:
        if input_data.channels is not None:
            raise ValueError("VDIFFormatter currently only supports unchannelised data")
        if samples_per_frame % (2 * VDIFEncode2Bit.SAMPLES_PER_WORD) != 0:
            raise ValueError(f"samples_per_frame must be a multiple of {2 * VDIFEncode2Bit.SAMPLES_PER_WORD}")
        self._input_it = iter(input_data)
        self._header = VDIFHeader.fromvalues(
            bps=2,
            complex_data=False,
            nchan=1,
            station=station,
            samples_per_frame=samples_per_frame,
            edv=0,
        )
        self._threads = threads
        words_per_frame = samples_per_frame // VDIFEncode2Bit.SAMPLES_PER_WORD
        frame_rate = 1 / (words_per_frame * input_data.time_scale)
        if frame_rate.denominator != 1:
            raise ValueError("samples_per_frame does not yield an integer frame rate")
        self._frame_rate = int(frame_rate)

        self.time_base = input_data.time_base
        self.time_scale = input_data.time_scale
        self.channels = None
        self.is_cupy = False

    def _frame(self, thread_id: int, header: VDIFHeader, frame_data: xr.DataArray) -> VDIFFrame:
        assert frame_data.dims == ("time",)
        header = header.copy()
        header.update(thread_id=thread_id)
        return VDIFFrame(
            header,
            VDIFPayload(frame_data.to_numpy(), header),
        )

    def _frame_set(self, frame_data: xr.DataArray) -> VDIFFrameSet:
        header = self._header.copy()

        time_offset = frame_data.attrs["time_bias"] * self.time_scale
        # Manually split time_offset (a Fraction) into seconds and fractions
        # of a second, to give a high-accuracy TimeDelta.
        time_offset_secs = int(time_offset)
        time_offset_frac = float(time_offset - time_offset_secs)
        time = self.time_base + TimeDelta(time_offset_secs, time_offset_frac, scale="tai", format="sec")
        header.set_time(time, frame_rate=self._frame_rate * u.Hz)

        frames = [self._frame(i, header, frame_data.sel(thread_idx)) for i, thread_idx in enumerate(self._threads)]
        return VDIFFrameSet(frames, header)

    def __iter__(self) -> Iterator[VDIFFrameSet]:
        words_per_frame = self._header.samples_per_frame // VDIFEncode2Bit.SAMPLES_PER_WORD
        for buffer in self._input_it:
            n_frames = buffer.sizes["time"] // words_per_frame
            for i in range(n_frames):
                word_start = i * words_per_frame
                word_stop = (i + 1) * words_per_frame
                frame_data = buffer.isel(time=np.s_[word_start:word_stop])
                frame_data.attrs["time_bias"] += word_start
                yield self._frame_set(frame_data)
