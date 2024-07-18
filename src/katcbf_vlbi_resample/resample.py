# Copyright (c) 2024, National Research Foundation (SARAO)

"""Resample to a different frequency and split into sidebands."""

from fractions import Fraction
from typing import Iterable, Iterator, Self

import numpy as np
import pandas as pd
import scipy.signal
import xarray as xr

from . import xrsig
from .parameters import ResampleParameters, StreamParameters


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


class IFFT:
    """Iterator adapter than converts frequency domain to time domain."""

    def __init__(self, input_data: Iterable[xr.DataArray], first_sample: int = 0) -> None:
        self.first_sample = first_sample
        self._input_it = iter(input_data)

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> xr.DataArray:
        while True:
            chunk = next(self._input_it)
            sample0 = int(chunk.coords["time"][0])  # Force to Python int
            n_channels = chunk.sizes["channel"]
            out_samples = chunk.sizes["time"] * n_channels
            if self.first_sample < sample0 + out_samples:
                break  # We've reached a valid chunk

        n_channels = chunk.sizes["channel"]
        # Move middle frequency to position 0
        chunk = chunk.roll(channel=-(n_channels // 2))
        # Convert back to time domain with inverse FFT
        chunk = xrsig.ifft(chunk, dim="channel")
        # The channel dimension is now flattened into a time dimension. We
        # have to use a temporary name for it, since xarray doesn't like
        # using the same name for the new dimension.
        chunk = chunk.drop_vars("time")
        chunk = chunk.stack(tmp=("time", "channel"), create_index=False)
        chunk = chunk.rename(tmp="time")
        chunk.coords["time"] = pd.RangeIndex(sample0, sample0 + out_samples)
        if self.first_sample > sample0:
            chunk = chunk.isel(time=np.s_[self.first_sample :])
        return chunk


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
    return out


class Resample:
    """Resample to a different frequency and split into sidebands.

    This class is used to wrap an iterator that yields input data in chunks
    to one that yields output data in chunks. The chunks are not one-to-one
    due to windowing effects.

    Each input axis must have a `time` dimension, and it must have
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
        input_data: Iterable[xr.DataArray],
    ) -> None:
        if input_params.bandwidth < output_params.bandwidth:
            raise ValueError("Cannot produce more output bandwidth than input")
        self._ratio = Fraction(output_params.bandwidth) / Fraction(input_params.bandwidth)
        n = self._ratio.numerator
        d = self._ratio.denominator
        self._fir = _fir_coeff_win(resample_params.fir_taps, resample_params.passband, self._ratio)
        # upfirdn does a full convolution, so we need to trim the result to get
        # only valid samples (e.g. see np.convolve). If we discard x output
        # samples, then the filter overhangs by taps - 1 - x*d, and this must
        # be strictly less than n.
        self._fir_discard_left = (resample_params.fir_taps - 1 - n) // d + 1
        self._hilbert = _hilbert_coeff_win(resample_params.hilbert_taps)
        mix_freq = input_params.center_freq - output_params.center_freq
        self._mix_scale = mix_freq / input_params.bandwidth
        self._input_it = iter(input_data)

    def __iter__(self) -> Iterator[xr.DataArray]:
        buffer = None
        n = self._ratio.numerator
        d = self._ratio.denominator
        for input_chunk in self._input_it:
            if buffer is None:
                buffer = input_chunk
            else:
                # TODO: check that it follows directly from previous chunk
                buffer = xr.concat([buffer, input_chunk], dim="time")
            # Determine first invalid sample. Output sample x takes taps up to
            # upsampled sample x*d (inclusive). This must be strictly less
            # than n*buffer_len for the sample to be valid.
            stop = (n * buffer.sizes["time"] - 1) // d + 1
            # Hilbert filter further eats into valid samples
            stop -= self._hilbert.sizes["time"] - 1
            # Additionally, we only want to take a number of samples that
            # leaves the phase of upfirdn intact for the next batch.
            stop -= (stop - self._fir_discard_left) % n
            if stop > self._fir_discard_left:
                mixed = buffer * np.exp(2j * np.pi * self._mix_scale * buffer.coords["time"])
                convolved = xrsig.upfirdn(
                    self._fir,
                    mixed,
                    up=self._ratio.numerator,
                    down=self._ratio.denominator,
                    dim="time",
                )
                convolved = _split_sidebands(convolved, self._hilbert)
                convolved = convolved.isel(time=np.s_[self._fir_discard_left : stop])
                # TODO: add time coords to output
                yield convolved
                # Number of input samples to advance
                used = int(convolved.sizes["time"] / self._ratio)
                buffer = buffer.isel(time=np.s_[used:])
