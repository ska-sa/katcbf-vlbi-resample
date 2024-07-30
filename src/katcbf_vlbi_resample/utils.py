# Copyright (c) 2024, National Research Foundation (SARAO)

"""Miscellaneous utilities."""

from collections.abc import Sequence

import xarray as xr


def concat_time(arrays: Sequence[xr.DataArray]) -> xr.DataArray:
    """Concatenate datasets in time.

    This checks that the arrays are contiguous in time and fixes up the
    timestamping attributes.
    """
    for attr in ["time_base", "time_scale"]:
        for array in arrays:
            if array.attrs[attr] != arrays[0].attrs[attr]:
                raise ValueError(f"Attribute {attr} is inconsistent")
    n = 0
    for array in arrays:
        if array.attrs["time_bias"] != arrays[0].attrs["time_bias"] + n:
            raise ValueError("Chunks are not contiguous in time")
        n += array.sizes["time"]
    return xr.concat(arrays, dim="time")
