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
- `channels` (:class:`int` | ``None``) — the number
  of channels in the stream, or ``None`` for unchannelised data
- `is_cupy` (:class:`bool`) — whether the chunks emitted from the stream
  contain cupy arrays

The meaning of `time_base` and `time_scale` is described below.

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

- `time_bias` (:class:`int`)

Consider a data element with index `i` on the `time` axis. Its timestamp is

.. code:: python

   time_base + TimeDelta(float((time_bias + i) * time_scale), scale="tai", format="sec")

The value ``time_bias + i`` is called the "sample index".
For channelised data, the associated timestamp should be the timestamp for the
first sample, and the length of the `channel` axis must match the stream's
`channels` attribute.

GPU acceleration
----------------
The GPU acceleration uses `cupy`_ to replace operations that would otherwise
have been done with numpy or scipy. To facilitate testing and the
:option:`!--cpu` command-line option, most stream classes support either cupy
or numpy arrays. In some cases the naïve approach of using the same code for
both backends was found to be slow, and cupy-specific codepaths (using custom
kernels) was used instead for better performance.

This approach is still sub-optimal, with many unnecessary copies and more
passes over the memory than necessary. However, it has the benefit of being
flexible, as the various stream classes can be stacked together in a variety
of ways.

One important optimisation is overlapping of CPU and GPU work. This is
achieved in the :class:`.AsNumpy` class, which transfers results from the GPU
to the CPU. It uses a non-blocking copy so that the CPU can proceed while the
GPU computes the results that have been requested, and keeps a queue of
transfers that are in flight.

It should be noted that transfer to and from the GPU are still serialised with
respect to the GPU computations, as there is only a single CUDA stream. These
transfers take very little time compared to the computations, so this would
have only a small benefit.

.. _cupy: https://docs.cupy.dev/
