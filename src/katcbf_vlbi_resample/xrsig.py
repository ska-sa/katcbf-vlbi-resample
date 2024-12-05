# Copyright (c) 2024, National Research Foundation (SARAO)

"""Wrap various scipy signal processing functions for use with xarray."""

import warnings
from collections.abc import Hashable

import cupy as cp
import cupyx.scipy.fft
import numpy as np
import scipy.fft
import scipy.signal
import xarray as xr

from .utils import is_cupy

# cupyx.scipy.signal reports a warning on import.
warnings.filterwarnings("ignore", message=r"cupyx\.jit\.rawkernel is experimental", category=FutureWarning)
import cupyx.scipy.signal  # noqa: E402


def _wrap_cupyx_upfirdn(h: cp.ndarray, x: cp.ndarray, *args, **kwargs):
    # Work around a few bugs:
    # https://github.com/cupy/cupy/issues/8448
    # https://github.com/cupy/cupy/issues/8449
    if kwargs.get("mode") == "constant":
        del kwargs["mode"]

    if x.ndim == 1:
        return cupyx.scipy.signal.upfirdn(h, x, *args, **kwargs)
    elif x.ndim == 2:
        return cp.stack([cupyx.scipy.signal.upfirdn(h, row, *args, **kwargs) for row in x])
    else:
        raise NotImplementedError("upfirdn not implemented for ndim > 2 yet")


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
        _wrap_cupyx_upfirdn if is_cupy(x) else scipy.signal.upfirdn,
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
    kwargs = dict(norm=norm, overwrite_x=overwrite_x, workers=workers, plan=plan)
    if is_cupy(x):
        del kwargs["workers"]  # not supported by cupyx.scipy.fft
    return xr.apply_ufunc(
        cupyx.scipy.fft.ifft if is_cupy(x) else scipy.fft.ifft,
        x,
        n,
        input_core_dims=[[dim], [dim]],
        output_core_dims=[(dim,)],
        exclude_dims={dim},
        kwargs=kwargs,
        keep_attrs=True,
    )


def _wrap_cupyx_convolve1d(in1: cp.ndarray, in2: cp.ndarray, *args, **kwargs) -> cp.ndarray:
    # cp.broadcast unfortunately doesn't allow the final (time) dimensions to
    # have different sizes, so we have to do more complicated things to figure
    # out the broadcast shape.
    pshape1 = in1.shape[:-1]
    pshape2 = in2.shape[:-1]
    if pshape1 != pshape2:
        bshape = np.broadcast_shapes(pshape1, pshape2)
        in1 = cp.broadcast_to(in1, bshape + in1.shape[-1:])
        in2 = cp.broadcast_to(in2, bshape + in2.shape[-1:])
    if in1.ndim == 1:
        return cupyx.scipy.signal.convolve(in1, in2, *args, **kwargs)
    else:
        return cp.stack([_wrap_cupyx_convolve1d(x, y, *args, **kwargs) for x, y in zip(in1, in2)])


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
    assert is_cupy(in1) == is_cupy(in2)
    use_cupy = is_cupy(in1)
    return xr.apply_ufunc(
        _wrap_cupyx_convolve1d if use_cupy else scipy.signal.convolve,
        in1,
        in2,
        vectorize=not use_cupy,  # Doesn't work for cupy
        input_core_dims=[[dim], [dim]],
        output_core_dims=[(dim,)],
        exclude_dims={dim},
        kwargs=dict(mode=mode, method=method),
    )
