# Copyright (c) 2024, National Research Foundation (SARAO)

"""Wrap various scipy signal processing functions for use with xarray."""

from collections.abc import Hashable

import scipy.fft
import scipy.signal
import xarray as xr


def upfirdn(
    h: xr.DataArray,
    x: xr.DataArray,
    up: int = 1,
    down: int = 1,
    *,
    dim: Hashable,
    mode: str = "constant",
    cval: float = 0.0,
) -> xr.DataArray:
    """Wrap :func:`scipy.signal.upfirdn` for :class:`xarray.DataArray`."""
    out = xr.apply_ufunc(
        scipy.signal.upfirdn,
        h,
        x,
        input_core_dims=[[dim], [dim]],
        output_core_dims=[(dim,)],
        exclude_dims={dim},
        keep_attrs="drop",
        kwargs=dict(up=up, down=down, mode=mode, cval=cval),
    )
    # Copy attrs from x, not h, which would be the default (being first)
    out.attrs = x.attrs
    return out


def ifft(
    x: xr.DataArray,
    n: int | None = None,
    *,
    dim: Hashable,
    norm: str = "backward",
    overwrite_x: bool = False,
    workers: int | None = None,
    plan=None,
) -> xr.DataArray:
    """Wrap :func:`scipy.fft.ifft` for :class:`xarray.DataArray`."""
    return xr.apply_ufunc(
        scipy.fft.ifft,
        x,
        n,
        input_core_dims=[[dim], [dim]],
        output_core_dims=[(dim,)],
        exclude_dims={dim},
        kwargs=dict(norm=norm, overwrite_x=overwrite_x, workers=workers, plan=plan),
        keep_attrs=True,
    )


def convolve1d(
    in1: xr.DataArray,
    in2: xr.DataArray,
    dim: Hashable,
    mode: str = "full",
    method: str = "auto",
) -> xr.DataArray:
    """Wrap :func:`scipy.signal.convolve` for :class:`xarray.DataArray`.

    It always performs 1D convolution, using the `dim` argument to select
    the axis over which to convolve.
    """
    return xr.apply_ufunc(
        scipy.signal.convolve,
        in1,
        in2,
        vectorize=True,
        input_core_dims=[[dim], [dim]],
        output_core_dims=[(dim,)],
        exclude_dims={dim},
        kwargs=dict(mode=mode, method=method),
    )
