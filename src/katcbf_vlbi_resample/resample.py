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

"""Signal processing algorithms."""

from collections.abc import Iterator
from fractions import Fraction

import cupy as cp
import numpy as np
import scipy.signal
import xarray as xr
from astropy.time import Time, TimeDelta

from . import xrsig
from .parameters import ResampleParameters, StreamParameters
from .stream import ChunkwiseStream, Stream
from .utils import as_cupy, concat_time, isel_time, time_align


def _time_delta_to_sample(dt: TimeDelta, time_scale: Fraction) -> int:
    """Convert a time offset to the nearest sample index."""
    # This implementation preserves the full precision of the TimeDelta
    # by using Fraction's arbitrary precision.
    time_scale_days = time_scale / 86400
    return round((Fraction(dt.jd1) + Fraction(dt.jd2)) / time_scale_days)


class ClipTime:
    """Iterator adapter that limits the selected time range.

    Note that it's generally more efficient if the upstream iterator
    can do the work. However, this adapter is convenient for clipping
    to sample accuracy after the upstream clips to chunk accuracy.

    The range to select is based on sample indices, as given by
    the `time_bias` attribute on chunks. Alternatively,
    one can select by absolute time by passing instances of
    :class:`astropy.time.Time`.
    """

    def __init__(
        self, input_data: Stream[xr.DataArray], start: Time | int | None = None, stop: Time | int | None = None
    ) -> None:
        if start is None or isinstance(start, int):
            self._start = start
        else:
            self._start = _time_delta_to_sample(start - input_data.time_base, input_data.time_scale)

        if stop is None or isinstance(stop, int):
            self._stop = stop
        else:
            self._stop = _time_delta_to_sample(stop - input_data.time_base, input_data.time_scale)

        self.time_base = input_data.time_base
        self.time_scale = input_data.time_scale
        self.channels = input_data.channels
        self.is_cupy = input_data.is_cupy
        self._input_it = iter(input_data)

    def __iter__(self) -> Iterator[xr.DataArray]:
        return self

    def __next__(self) -> xr.DataArray:
        while True:
            chunk = next(self._input_it)
            chunk_start: int = chunk.attrs["time_bias"]
            chunk_stop = chunk_start + chunk.sizes["time"]
            start = chunk_start if self._start is None else self._start
            stop = chunk_stop if self._stop is None else self._stop
            if chunk_start >= stop:
                raise StopIteration  # We've gone past `stop`
            if chunk_stop > start:
                break  # We've found a chunk which is at least partially kept

        if start > chunk_start or stop < chunk_stop:
            slice_start = max(0, start - chunk_start)
            slice_stop = min(chunk.sizes["time"], stop - chunk_start)
            chunk = isel_time(chunk, np.s_[slice_start:slice_stop])
        return chunk


class IFFT(ChunkwiseStream[xr.DataArray, xr.DataArray]):
    """Iterator adapter that converts frequency domain to time domain."""

    def __init__(self, input_data: Stream[xr.DataArray]) -> None:
        if input_data.channels is None:
            raise TypeError("A stream with channels is required as input")
        super().__init__(input_data)
        self.time_scale = input_data.time_scale / Fraction(input_data.channels)
        self.channels = None
        self.is_cupy = input_data.is_cupy
        self._in_channels = input_data.channels

    def _transform(self, chunk: xr.DataArray) -> xr.DataArray:
        assert chunk.sizes["channel"] == self._in_channels
        # Move middle frequency to position 0
        chunk = chunk.roll(channel=-(self._in_channels // 2))
        # Convert back to time domain with inverse FFT. The roll necessarily
        # makes a copy, so it's safe to overwrite the input.
        chunk = xrsig.ifft(chunk, dim="channel", norm="ortho", overwrite_x=True)
        # The channel dimension is now flattened into a time dimension. We
        # have to use a temporary name for it, since xarray doesn't like
        # using the same name for the new dimension.
        chunk = chunk.stack(tmp=("time", "channel"), create_index=False)
        chunk = chunk.rename(tmp="time")
        chunk.attrs["time_bias"] *= self._in_channels
        return chunk


def _fir_coeff_win(taps: int, passband: float, ratio: Fraction, window: str = "hamming") -> xr.DataArray:
    """Calculate coefficients for upfirdn filter."""
    fir = scipy.signal.firwin(
        taps,
        [0.5 - 0.5 * passband, 0.5 + 0.5 * passband],
        pass_zero="bandpass",
        window=window,
        fs=2 * ratio.denominator,
    )
    # Normalise for unitary incoherent gain
    fir *= np.sqrt(ratio.numerator / np.sum(np.square(fir)))
    return xr.DataArray(fir, dims=("time",)).astype(np.float32)


def _upfirdn(h: xr.DataArray, x: xr.DataArray, ratio: Fraction) -> xr.DataArray:
    """Apply upfirdn operation and adjust timestamp attributes."""
    x = xrsig.upfirdn(
        h,
        x,
        up=ratio.numerator,
        down=ratio.denominator,
        dim="time",
    )
    # Emulate upsampling
    x.attrs["time_bias"] *= ratio.numerator
    # Shift to centre the filter (upfirdn is a "full" convolution, hence
    # the reference point moves backwards in time).
    x.attrs["time_bias"] -= h.sizes["time"] // 2
    # Downsample
    assert x.attrs["time_bias"] % ratio.denominator == 0
    x.attrs["time_bias"] //= ratio.denominator
    return x


def _hilbert_coeff_win(N: int, window: str = "hamming") -> xr.DataArray:
    """Calculate Hilbert filter coefficients using window-based design.

    Computes the ideal Hilbert filter impulse response, as per Proakis &
    Monolakis, 3rd Edition, p. 659, eq 8.2.86, cropped to the window size. Then
    scales the impulse response by the specified window.

    Parameters
    ----------
    N
        Number of filter taps. Must be odd.
    window
        Window description as accepted by :func:`scipy.signal.get_window`.
    """
    assert N % 2 == 1
    # Define symmetric ideal impulse response out to the window length
    nn_n = np.arange(-(N // 2), 0)
    nn_p = np.arange(1, N // 2 + 1)
    hd_n = 2 * np.sin(np.pi * nn_n / 2.0) ** 2 / (np.pi * nn_n)
    hd_p = 2 * np.sin(np.pi * nn_p / 2.0) ** 2 / (np.pi * nn_p)
    hd = np.r_[hd_n, 0, hd_p]
    # Compute coefficients for a symmetric window as specified
    win_coeff = scipy.signal.get_window(window, N, fftbins=False)
    # Return windowed impulse response
    return xr.DataArray(hd * win_coeff, dims=("time",)).astype(np.float32)


def _split_sidebands(data: xr.DataArray, coeff: xr.DataArray) -> xr.DataArray:
    """Split a complex signal into lower and upper sidebands.

    Parameters
    ----------
    data
        Input data, with a `time` axis
    coeff
        Coefficients for Hilbert filter (see :func:`_hilbert_coeff_win`) with an
        odd number of taps

    Returns
    -------
    sidebands
        An array of real data, with an extra `sideband` dimension with labels
        `lsb` and `usb`.
    """
    h = xrsig.convolve1d(data.imag, coeff, dim="time", mode="valid")
    # Trim the edges of 'data' to make it match up with h
    trim = coeff.sizes["time"] // 2
    data = isel_time(data, np.s_[trim:-trim])
    lsb = data.real + h
    usb = data.real - h
    out = xr.concat(
        [lsb, usb],
        dim="sideband",
        coords="minimal",
    )
    out.coords["sideband"] = ["lsb", "usb"]
    out.attrs = data.attrs
    return out


def _mixer(buffer: xr.DataArray, scale: float, start_cycles: float) -> xr.DataArray:
    if isinstance(buffer.data, cp.ndarray):
        kernel = cp.ElementwiseKernel(
            "float64 scale, float64 start_cycles",
            "complex64 out",
            """
            double cycles = i * scale + start_cycles;
            cycles -= rint(cycles);
            float s, c;
            sincosf((float) cycles * (2 * (float) M_PI), &s, &c);
            out = complex<float>(c, s);
            """,
            "mixer",
        )
        return kernel(scale, start_cycles, size=buffer.sizes["time"])
    else:
        cycles = np.arange(buffer.sizes["time"], dtype=np.float64) * scale + start_cycles
        # Remove integer part before dropping to single precision, to reduce rounding
        # issues.
        cycles -= np.rint(cycles)
        cycles = cycles.astype(np.float32)
        return xr.DataArray(np.exp(2j * np.pi * cycles), dims=("time",))


class Resample:
    """Resample to a different frequency and split into sidebands.

    This class is used to wrap an iterator that yields input data in chunks
    to one that yields output data in chunks. The chunks are not one-to-one
    due to windowing effects.

    Each input array must have a `time` dimension, and it must have
    a coordinate that contains consecutive integers. Each subsequent
    chunk must also follow on contiguously from the previous one. Chunks need
    not be the same size. Input samples are expected to be complex values,
    and the sampling rate equals the bandwidth.

    The output has an additional `sideband` dimension to indicate the lower
    (`lsb`) or upper (`usb`) sideband. Output samples are real.
    """

    def __init__(
        self,
        input_params: StreamParameters,
        output_params: StreamParameters,
        resample_params: ResampleParameters,
        input_data: Stream[xr.DataArray],
    ) -> None:
        if input_params.bandwidth < output_params.bandwidth:
            raise ValueError("Cannot produce more output bandwidth than input")
        if Fraction(input_params.bandwidth) * input_data.time_scale != 1:
            # TODO: just eliminate bandwidth from StreamParameters?
            raise ValueError("Input bandwidth is not consistent with the input stream")
        self._ratio = Fraction(output_params.bandwidth) / Fraction(input_params.bandwidth)
        n, d = self._ratio.as_integer_ratio()
        self._fir = _fir_coeff_win(resample_params.fir_taps, resample_params.passband, self._ratio)
        # upfirdn does a full convolution, so we need to trim the result to get
        # only valid samples (e.g. see np.convolve). If we discard x output
        # samples, then the filter overhangs by taps - 1 - x*d, and this must
        # be strictly less than n.
        self._fir_discard_left = (resample_params.fir_taps - 1 - n) // d + 1
        # We need the time_bias calculation in _upfirdn to produce an integer
        # output. That happens when t * n - fir_taps // 2 is divisible by d.
        self._time_bias_mod = resample_params.fir_taps // 2 * pow(n, -1, mod=d) % d
        self._hilbert = _hilbert_coeff_win(resample_params.hilbert_taps)
        self._mix_freq = input_params.center_freq - output_params.center_freq
        self._input_it = iter(input_data)
        self._in_time_scale = input_data.time_scale
        self.time_base = input_data.time_base
        self.time_scale = input_data.time_scale / self._ratio
        self.channels = input_data.channels
        self.is_cupy = input_data.is_cupy
        if input_data.is_cupy:
            # Transfer filters to the GPU once
            self._fir = as_cupy(self._fir, blocking=True)
            self._hilbert = as_cupy(self._hilbert, blocking=True)

    def __iter__(self) -> Iterator[xr.DataArray]:
        buffer = None
        n, d = self._ratio.as_integer_ratio()
        for input_chunk in self._input_it:
            if buffer is None:
                # TODO: could instead pad on the left, then discard invalid
                # samples, which would avoid discarding good data.
                buffer = time_align(input_chunk, d, self._time_bias_mod)
                if buffer is None:
                    continue  # We've trimmed the entire input chunk
            else:
                buffer = concat_time([buffer, input_chunk])
            n_time = buffer.sizes["time"]
            # Determine first invalid sample. Output sample x takes taps up to
            # upsampled sample x*d (inclusive). This must be strictly less
            # than n*n_time for the sample to be valid.
            stop = (n * n_time - 1) // d + 1
            # Hilbert filter further eats into valid samples
            stop -= self._hilbert.sizes["time"] - 1
            # Additionally, we only want to take a number of samples that
            # leaves the phase of upfirdn intact for the next batch.
            stop -= (stop - self._fir_discard_left) % n
            if stop > self._fir_discard_left:
                mix_scale = Fraction(self._mix_freq) * self._in_time_scale
                # Compute the phase for the first sample using Fraction to avoid
                # rounding errors creeping in when time_bias is large.
                start_cycles = mix_scale * buffer.attrs["time_bias"]
                start_cycles -= round(start_cycles)
                mixer = _mixer(buffer, float(mix_scale), float(start_cycles))
                mixed = buffer * mixer
                mixed.attrs = buffer.attrs
                convolved = _upfirdn(self._fir, mixed, self._ratio)
                convolved = _split_sidebands(convolved, self._hilbert)
                convolved = isel_time(convolved, np.s_[self._fir_discard_left : stop])
                yield convolved
                # Number of input samples to advance
                assert (convolved.sizes["time"] / self._ratio).denominator == 1
                used = int(convolved.sizes["time"] / self._ratio)
                buffer = isel_time(buffer, np.s_[used:])
