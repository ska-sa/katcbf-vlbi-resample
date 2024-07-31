Design
======

Iterators
---------
The stream of sample values is broken up into a chunks, and is presented as a
Python iterator that yields the chunks one at a time. This serial, ordered
approach is in contrast to frameworks like Dask_ that allow random and
parallel access to all the chunks. That model is powerful for offline
processing, but an iterator-based design more closely models an online
processing pipeline.

.. _Dask: https://docs.dask.org/en/stable/

Some metadata is associated with the sample stream rather than the individual
samples. Each stream class must implement the :class:`.stream.Stream`
protocol, which requires the following attributes:

- `time_base` (:class:`astropy.time.Time`)
- `time_scale` (:class:`fraction.Fraction`)
- `channels` (:class:`int`, or ``None`` for unchannelised data)

Their meaning is described below.

Chunk format
------------
Each chunk is an :class:`xarray.DataArray` with a set of conventional
dimensions and attributes. For efficiency, coordinates are not always used.
The following dimensions are recognised:

- `time` should always be present
- `channel` should be present for channelised data
- `pol`, with coordinates that label the polarisations e.g., as ``h`` and
  ``v``.
- `sideband`, with coordinates that label the sidebands as ``lsb`` and
  ``usb``.

To allow for arithmetic on timestamps without rounding errors, a somewhat
complex scheme is employed. In addition to the stream attributes listed
above, each chunk has the following xarray attribute:

- `time_bias` (:class:`fraction.Fraction`)

Consider a data element with index `i` on the `time` axis. Its timestamp is

.. code:: python

   time_base + TimeDelta(float((time_bias + i) * time_scale), scale="tai", format="sec")

The value ``time_bias + i`` is called the "sample index".
For channelised data, the associated timestamp should be the timestamp for the
first sample, and the length of the `channel` axis must match the stream's
`channels` attribute.
