################################################################################
# Copyright (c) 2024-2026, National Research Foundation (SARAO)
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

"""Encode data to VDIF frames."""

from collections.abc import AsyncIterator
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Final

import cupy as cp
import numpy as np
import xarray as xr
from astropy.time import Time, TimeDelta

from .stream import Stream
from .utils import concat_time, isel_time, time_align


# Note: the caller must pass the inverse of the threshold here. Computing
# the inverse inside this function would cause it to happen per-element on
# the GPU (much more expensive) rather than per-array in the Python code.
@cp.fuse
def _encode_2bit_prep(values, inverse_threshold: float):
    xp = cp.get_array_module(values)
    return xp.clip(values * inverse_threshold + 2, 0, 3).astype(np.uint8)


def _encode_2bit(values, threshold: float):
    """Encode values using 2 bits per value, packing the result into bytes.

    This is equivalent to :func:`baseband.vdif.encode_2bit` but it supports a
    custom threshold and cupy. Note that values very close to the threshold
    might round differently.
    """
    xp = cp.get_array_module(values)
    values = _encode_2bit_prep(values, 1.0 / threshold)
    values = values.reshape(values.shape[:-1] + (-1, 4))
    values <<= xp.arange(0, 8, 2, dtype=np.uint8)
    # cupy doesn't currently support np.bitwise_or.reduce
    return values[..., 0] | values[..., 1] | values[..., 2] | values[..., 3]


def _encode_2bit_words(values, threshold: float):
    return _encode_2bit(values, threshold).view("<u4")


def _reference_epoch(time: Time) -> tuple[Time, int]:
    """Compute the VDIF reference epoch to use.

    VDIF has a reference epoch every 6 months, with 0 corresponding to
    00:00:00 UTC on 1 Jan 2000. This function will pick the latest
    reference epoch prior to (or equal to) `time`.

    Returns
    -------
    ref_time
        The UTC time corresponding to the chosen reference epoch.
    epoch
        Index of the epoch. It is *not* truncated to 6 bits.
    """
    t = time.utc.ymdhms
    if t.year < 2000:
        raise ValueError("Times before 2000 cannot be used with VDIF")
    epoch = (t.year - 2000) * 2
    ref_month = 1
    if t.month >= 7:
        epoch += 1
        ref_month = 7
    return Time(dict(year=t.year, month=ref_month, day=1), scale="utc"), epoch


class VDIFEncode2Bit:
    """Quantise and pack values to 2-bit samples.

    Values are quantised by comparing the absolute value to `threshold`.
    The power levels must already have been adjusted to make this an
    threshold appropriate.

    The output array is in units of VDIF words (32-bit little-endian) rather
    than samples, and `time_bias` is also in units of words. Only real-valued
    unchannelised data is supported.

    The input need not have any particular alignment. The output is aligned
    to the VDIF frame length. This also changes the `time_base` and introduces
    up to half a sample of delay (positive or negative) to align to the VDIF
    sample clock. The amount of the delay depends only on the incoming
    `time_base` and `time_scale`.
    """

    SAMPLES_PER_WORD: Final = 16

    def __init__(self, input_data: Stream[xr.DataArray], samples_per_frame: int, threshold: float) -> None:
        if input_data.channels is not None:
            raise ValueError("VDIFEncoder2Bit currently only supports unchannelised data")
        # VDIF requires an even number of words per frame
        if samples_per_frame % (2 * self.SAMPLES_PER_WORD) != 0:
            raise ValueError(f"samples_per_frame must be a multiple of {2 * self.SAMPLES_PER_WORD}")
        if (samples_per_frame * input_data.time_scale).numerator != 1:
            raise ValueError("samples_per_frame does not yield an integer frame rate")

        self._input_it = aiter(input_data)

        # Shift the time_base back to the last whole second. Note that astropy
        # times are only accurate to around 15ps, but we're likely introducing
        # larger errors in the rounding of _shift_samples below.
        shift = input_data.time_base.utc.ymdhms.second % 1
        #: Amount to add to incoming time_bias
        self._shift_samples = round(Fraction(shift) / input_data.time_scale)

        self.samples_per_frame = samples_per_frame
        self.time_base = input_data.time_base - TimeDelta(shift, format="sec", scale="tai")
        self.time_scale = input_data.time_scale * self.SAMPLES_PER_WORD
        self.channels = None
        self.is_cupy = input_data.is_cupy
        self.threshold = threshold

    async def __aiter__(self) -> AsyncIterator[xr.DataArray]:
        samples_per_frame = self.samples_per_frame
        buffer = None
        async for in_data in self._input_it:
            in_data.attrs["time_bias"] += self._shift_samples
            if buffer is None:
                # Discard leading partial frame.
                buffer = time_align(in_data, samples_per_frame, 0)
                if buffer is None:
                    continue  # Don't have enough data to reach the frame boundary
            else:
                buffer = concat_time([buffer, in_data])
            del in_data

            n_frames = buffer.sizes["time"] // samples_per_frame
            encoded = xr.apply_ufunc(
                _encode_2bit_words,
                buffer.isel(time=np.s_[: n_frames * samples_per_frame]),
                input_core_dims=[["time"]],
                output_core_dims=[("time",)],
                exclude_dims={"time"},
                keep_attrs=True,
                kwargs=dict(threshold=self.threshold),
            )
            assert encoded.attrs["time_bias"] % self.SAMPLES_PER_WORD == 0
            encoded.attrs["time_bias"] //= self.SAMPLES_PER_WORD
            yield encoded
            del encoded
            # Cut off the piece that's been processed. We then copy the
            # tail piece so that the bulk of the memory can be freed.
            skip = n_frames * samples_per_frame
            buffer = isel_time(buffer, np.s_[skip:]).copy()


@dataclass
class VDIFFrame:
    """Combines a VDIF header and payload in a data class."""

    header: np.ndarray
    payload: np.ndarray


def _make_header(
    *,
    bps: int,
    station: str | int,
    samples_per_frame: int,
    ref_time: Time,
) -> tuple[np.ndarray, Time]:
    """Generate a template of the VDIF header, without per-frame timing data."""

    def field(name: str, word: int, offset: int, bits: int, value: int) -> None:
        assert offset + bits <= 32
        if value >= 1 << bits:
            raise ValueError(f"{name} does not fit in the {bits}-bit VDIF header field")
        header[word] |= value << offset

    header = np.zeros(8, "<u4")
    ref_time, epoch = _reference_epoch(ref_time)
    epoch &= 0x3F  # VDIF only uses the bottom 6 bits, wrapping in 2032
    field("ref epoch", 1, 24, 6, epoch)
    if samples_per_frame * bps % 64 != 0:
        raise ValueError("samples_per_frame does not yield a whole number of 8-byte units")
    field("data frame length", 2, 0, 24, samples_per_frame * bps // 64 + 4)
    field("version", 2, 29, 3, 1)
    if isinstance(station, int):
        if station < 0 or station >= 0x3000:
            raise ValueError("numeric station ID is out of range")
        field("station ID", 3, 0, 16, station)
    else:
        # TODO: check that it's ASCII and the first character is >= '0'
        field("station ID", 3, 0, 16, (ord(station[0]) << 8) | ord(station[1]))
    field("bits/sample", 3, 26, 5, bps - 1)
    return header, ref_time


class VDIFFormatter:
    """Encode data to VDIF frames.

    The data must have already been quantised to words and aligned to frames
    using :class:`VDIFEncode2Bit` or similar. The time_base must be on a second
    boundary.

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
        self._input_it = aiter(input_data)
        self._header, ref_time = _make_header(
            bps=2,
            station=station,
            samples_per_frame=samples_per_frame,
            ref_time=input_data.time_base,
        )
        self._samples_per_frame = samples_per_frame
        self._threads = threads
        words_per_frame = samples_per_frame // VDIFEncode2Bit.SAMPLES_PER_WORD
        frame_rate = 1 / (words_per_frame * input_data.time_scale)
        if frame_rate.denominator != 1:
            raise ValueError("samples_per_frame does not yield an integer frame rate")
        self._frame_rate = int(frame_rate)
        #: Number of VDIF seconds corresponding to a time_bias of 0
        base_seconds = (input_data.time_base - ref_time).sec
        self._base_seconds = round(base_seconds)
        if abs(base_seconds - self._base_seconds) > 1e-8:
            raise ValueError("time_base was not aligned to a second boundary")

        self.time_base = input_data.time_base
        self.time_scale = input_data.time_scale
        self.channels = None
        self.is_cupy = False

    def _frame(self, thread_id: int, header: np.ndarray, frame_data: np.ndarray) -> VDIFFrame:
        header = header.copy()
        header[3] |= thread_id << 16
        return VDIFFrame(header, frame_data)

    def _frame_set(self, frame: int, frame_data: list[np.ndarray]) -> list[VDIFFrame]:
        header = self._header.copy()
        header[0] = self._base_seconds + frame // self._frame_rate
        header[1] |= frame % self._frame_rate
        return [self._frame(i, header, thread) for i, thread in enumerate(frame_data)]

    async def __aiter__(self) -> AsyncIterator[list[VDIFFrame]]:
        words_per_frame = self._samples_per_frame // VDIFEncode2Bit.SAMPLES_PER_WORD
        async for buffer in self._input_it:
            n_frames = buffer.sizes["time"] // words_per_frame
            # xarray's overheads are too high to use it on a per-frame basis.
            # Turn the buffer into a plain ol' numpy array (thread × time).
            raw_data = [buffer.sel(thread_idx).to_numpy() for thread_idx in self._threads]
            assert all(data.ndim == 1 for data in raw_data)
            start_frame = int(buffer[0].attrs["time_bias"] * self._frame_rate * self.time_scale)
            del buffer
            for i in range(n_frames):
                word_start = i * words_per_frame
                word_stop = (i + 1) * words_per_frame
                time_idx = np.s_[word_start:word_stop]
                frame_data = [data[time_idx] for data in raw_data]
                yield self._frame_set(start_frame + i, frame_data)
            del raw_data
