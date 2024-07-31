# Copyright (c) 2024, National Research Foundation (SARAO)

"""Signal processing algorithms."""

from collections.abc import Iterator
from fractions import Fraction
from typing import Self

import numpy as np
import scipy.signal
import xarray as xr

from . import xrsig
from .parameters import ResampleParameters, StreamParameters
from .stream import Stream
from .utils import concat_time


class ClipTime:
    """Iterator adapter that limits the selected time range.

    Note that it's generally more efficient if the upstream iterator
    can do the work. However, this adapter is convenient for clipping
    to sample accuracy after the upstream clips to chunk accuracy.

    The range to select is based on sample indices, as given by
    the `time_bias` attribute on chunks. Note that it will only
    clip to a whole sample if `time_bias` is non-integral.
    """

    def __init__(self, input_data: Stream[xr.DataArray], start: int, stop: int) -> None:
        self._start = start
        self._stop = stop
        self.time_base = input_data.time_base
        self.time_scale = input_data.time_scale
        self.channels = input_data.channels
        self._input_it = iter(input_data)

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> xr.DataArray:
        while True:
            chunk = next(self._input_it)
            chunk_start = int(chunk.attrs["time_bias"])
            chunk_stop = chunk_start + chunk.sizes["time"]
            if chunk_start >= self._stop:
                raise StopIteration  # We've gone past `stop`
            if chunk_stop > self._start:
                break  # We've found a chunk which is at least partially kept

        if self._start > chunk_start or self._stop < chunk_stop:
            slice_start = max(0, self._start - chunk_start)
            slice_stop = min(chunk.sizes["time"], self._stop - chunk_start)
            chunk = chunk.isel(time=np.s_[slice_start:slice_stop])
            chunk.attrs["time_bias"] += slice_start
        return chunk


class IFFT:
    """Iterator adapter that converts frequency domain to time domain."""

    def __init__(self, input_data: Stream[xr.DataArray]) -> None:
        if input_data.channels is None:
            raise TypeError("A stream with channels is required as input")
        self.time_base = input_data.time_base
        self.time_scale = input_data.time_scale / Fraction(input_data.channels)
        self.channels = None
        self._in_channels = input_data.channels
        self._input_it = iter(input_data)

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> xr.DataArray:
        chunk = next(self._input_it)
        assert chunk.sizes["channel"] == self._in_channels
        # Move middle frequency to position 0
        chunk = chunk.roll(channel=-(self._in_channels // 2))
        # Convert back to time domain with inverse FFT
        chunk = xrsig.ifft(chunk, dim="channel")
        # The channel dimension is now flattened into a time dimension. We
        # have to use a temporary name for it, since xarray doesn't like
        # using the same name for the new dimension.
        chunk = chunk.stack(tmp=("time", "channel"), create_index=False)
        chunk = chunk.rename(tmp="time")
        chunk.attrs["time_bias"] *= self._in_channels
        return chunk


# TODO: switch to float32
def _fir_coeff_win(taps: int, passband: float, ratio: Fraction, window: str = "hamming") -> xr.DataArray:
    """Calculate coefficients for upfirdn filter."""
    fir = scipy.signal.firwin(
        taps,
        [1 - passband, passband],
        pass_zero="bandpass",
        window=window,
        fs=2 * ratio.denominator,
    )
    # Normalise for unitary incoherent gain
    fir *= np.sqrt(ratio.numerator / np.sum(np.square(fir)))
    return xr.DataArray(fir, dims=("time",))


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
    # Downsample, being careful to ensure that time_bias is
    # a Fraction.
    x.attrs["time_bias"] /= Fraction(ratio.denominator)
    return x


# TODO: switch to float32
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
    return xr.DataArray(hd * win_coeff, dims=("time",))


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
    data = data.isel(time=np.s_[trim:-trim])
    lsb = data.real + h
    usb = data.real - h
    out = xr.concat(
        [lsb, usb],
        dim="sideband",
        coords="minimal",
    )
    out.coords["sideband"] = ["lsb", "usb"]
    out.attrs = data.attrs
    out.attrs["time_bias"] += trim
    return out


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
        self._hilbert = _hilbert_coeff_win(resample_params.hilbert_taps)
        self._mix_freq = input_params.center_freq - output_params.center_freq
        self._input_it = iter(input_data)
        self._in_time_scale = input_data.time_scale
        self.time_base = input_data.time_base
        self.time_scale = input_data.time_scale / self._ratio
        self.channels = input_data.channels

    def __iter__(self) -> Iterator[xr.DataArray]:
        buffer = None
        n, d = self._ratio.as_integer_ratio()
        for input_chunk in self._input_it:
            if buffer is None:
                buffer = input_chunk
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
                # Now it's safe to drop to floating point
                cycles = np.arange(n_time) * float(mix_scale) + float(start_cycles)
                mixer = xr.DataArray(np.exp(2j * np.pi * cycles), dims=("time",))
                mixed = buffer * mixer
                mixed.attrs = buffer.attrs
                convolved = _upfirdn(self._fir, mixed, self._ratio)
                convolved = _split_sidebands(convolved, self._hilbert)
                convolved = convolved.isel(time=np.s_[self._fir_discard_left : stop])
                convolved.attrs["time_bias"] += self._fir_discard_left
                yield convolved
                # Number of input samples to advance
                assert (convolved.sizes["time"] / self._ratio).is_integer()
                used = int(convolved.sizes["time"] / self._ratio)
                buffer = buffer.isel(time=np.s_[used:])
                buffer.attrs["time_bias"] += used
