# Copyright (c) 2024, National Research Foundation (SARAO)

"""Miscellaneous utilities."""

from collections.abc import Sequence

import cupy as cp
import xarray as xr


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
