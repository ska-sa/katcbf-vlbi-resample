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

"""Miscellaneous utilities."""

import asyncio
import functools
from collections.abc import Sequence
from fractions import Fraction

import cupy as cp
import xarray as xr
from astropy.time import TimeDelta


def concat_time(arrays: Sequence[xr.DataArray]) -> xr.DataArray:
    """Concatenate datasets in time.

    This checks that the arrays are contiguous in time and fixes up the
    timestamping attribute.
    """
    n = 0
    for array in arrays:
        if array.attrs["time_bias"] != arrays[0].attrs["time_bias"] + n:
            raise ValueError("Chunks are not contiguous in time")
        n += array.sizes["time"]
    # Simply using xr.concat is slow because it builds an index on the
    # concatenated axis. This is hacky and will probably lose coordinates
    # in the general case, but is sufficient for the uses in this library.
    xp = cp.get_array_module(arrays[0].data)
    return xr.apply_ufunc(
        lambda *arrays: xp.concatenate(arrays, axis=-1),
        *arrays,
        input_core_dims=[["time"] for _ in arrays],
        output_core_dims=[("time",)],
        exclude_dims={"time"},
        keep_attrs=True,
    )


def isel_time(array: xr.DataArray, index: slice) -> xr.DataArray:
    """Slice an array along the time axis.

    This takes care of adjusting the `time_bias` attribute of the return value.
    The slice must have a step size of 1.
    """
    start, stop, step = index.indices(array.sizes["time"])
    if step != 1:
        raise ValueError("isel_time requires step = 1")
    array = array.isel(time=index)
    array.attrs["time_bias"] += start
    return array


def time_align(array: xr.DataArray, modulus: int, remainder: int = 0) -> xr.DataArray | None:
    """Trim the start of an array to ensure alignment in time.

    The array is trimmed such that the ``time_bias`` of the output satisfies
    ``time_bias % modulus == remainder % modulus``. If it is not possible to
    do so with a non-empty array, return ``None`` instead.

    If no trimming is required, the original array is returned (not a view).
    """
    assert 0 <= remainder < modulus
    time_bias: int = array.attrs["time_bias"]
    trim = (remainder - time_bias) % modulus
    if trim >= array.sizes["time"]:
        return None
    elif trim == 0:
        return array
    else:
        return isel_time(array, slice(trim, None, None))


def is_cupy(array: xr.DataArray) -> bool:
    """Determine whether `array` contains a cupy array."""
    return isinstance(array.data, cp.ndarray)


def as_cupy(array: xr.DataArray, blocking: bool = False) -> xr.DataArray:
    """Convert `array` to hold a cupy array if necessary."""
    return array.copy(data=cp.asarray(array.data, blocking=blocking))


def stream_future[T](value: T, stream: cp.cuda.Stream | None = None) -> asyncio.Future[T]:
    """Return a Future that is made ready when work on the current CUDA stream is complete."""
    loop = asyncio.get_running_loop()
    future = loop.create_future()
    target_stream = cp.cuda.get_current_stream() if stream is None else stream
    target_stream.launch_host_func(loop.call_soon_threadsafe, functools.partial(future.set_result, value))
    return future


def fraction_to_time_delta(secs: Fraction, scale: str = "tai") -> TimeDelta:
    """Convert a fraction in seconds to a TimeDelta."""
    days = secs / 86400
    # To get maximum accuracy, split into integer and fractional parts and
    # convert them separately to float.
    days_int = round(days)
    return TimeDelta(days_int, float(days - days_int), format="jd", scale=scale)
