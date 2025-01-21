Introduction
============
This package contains tools for resampling a complex-valued voltage time
series to a given rate and writing the result to a `VDIF`_ file or network
stream. It is designed for use with MeerKAT's narrowband beamformer output,
but the underlying classes are somewhat generic. The design is based on a
Jupyter notebook written by Marcel Gouws.

The code is accelerated using CUDA.

.. _VDIF: https://vlbi.org/vlbi-standards/vdif/

Installation
------------
Run ``pip install -r requirements.txt .``

This will install `cupy`_, which will take a while and might fail if you don't
already have CUDA installed. At least Python 3.11 is required.

.. _cupy: https://docs.cupy.dev/

Running
-------
An example command line is shown:

.. code-block:: sh

   mk_vlbi_resample \
       1698332059_narrow1_tied_array_channelised_voltage_0y.h5 \
       1698332059_narrow1_tied_array_channelised_voltage_0x.h5 \
       'test_vdif_{file_nr:08d}.vdif' \
       --telstate=1698332059_sdp_l0.full.rdb \
       --bandwidth 64e6 \
       --frequency 1626490000.0 \
       --fir-taps 7201 \
       --start 2023-10-26T14:56:00.000002084 \
       --duration 2.0

This specifies:

- The input single-polarisation beams, in the order they should appear in the
  output (in this case, vertical then horizontal).
- The output file pattern, where :samp:`{file_nr:08d}` will be replaced by the
  sequential file number padded to 8 digits. A new file is started every
  256 MiB (can be overridden with :option:`!--file-size`).
- The telescope state for the capture block to be converted.
- The total bandwidth (in Hz) at which to resample the signal (lower and
  upper sidebands together).
- The desired centre frequency of the output (in Hz).
- The number of taps for the rational resampling filter.
- The timestamp of the first sample to process, and the amount of data to
  process. These are approximate, as there are various alignment
  restrictions (particularly for VDIF frames) that will cause slightly more
  or less data to be used.

There are other command-line options. Run ``mk_vlbi_resample --help`` for
further details.

Polarisation conversion
-----------------------
By default, the output polarisations are the same as the inputs, but there is
support to change the polarisation basis. The description below assumes that
the polarisations are in a celestial frame (i.e., that parallactic angle
correction has been done), but if it has not this will simply carry through to
the output.

The polarisations of the input and output are specified with a command line
option :samp:`--polarisation={A},{B}:{C},{D}` where A and B are the
polarisations of the two inputs (in the order given) and C and D are the
output polarisation (in the order they will be assigned VDIF thread IDs). The
valid values for A, B, C and D are:

- :samp:`x`, :samp:`y` for linear polarisation (North and East respectively);
- :samp:`R`, :samp:`L` for circular polarisation; or
- any of the above with a :samp:`-` prefix to indicate that the values are
  negated (for example, :samp:`-x` is South), or a :samp:`+` prefix which has
  no effect.

Operation
---------
The :program:`mk_vlbi_resample` script performs the following steps:

- Channelised samples are loaded from the HDF5 file and aligned in time
  between the polarisations.
- Each spectrum is inverse Fourier transformed to recover time-domain data.
- The time-domain data is clipped to the selected start time.
- A mixer and a bandpass filter with a rational resampling factor are used to
  reduce the bandwidth and adjust the centre frequency.
- If requested, the polarisation basis is changed.
- The signal is split into positive and negative frequencies, which become
  the upper and lower side-bands, with only the real component retained.
- The power is normalised. The data is chunked in time and the values within
  each chunk are divided by their root-mean-square (RMS). The RMS is
  calculated independently for each stream (upper and lower sideband for each
  polarisation). The chunk size is implementation-dependent, and the whole
  algorithm may change in future versions.
- The samples are quantised to 2 bits and encoded as VDIF frames.

The output file contains four VDIF threads: two polarisations each with two
sidebands.

Limitations
-----------
The following features from Marcel's notebook are not currently implemented in
the script:

- Power normalisation is done independently on each chunk, whereas the
  notebook uses a sliding window to smooth the changes.
- The user is entirely responsible for computing timestamp corrections to
  apply.
- There is no support for conversion to a circular polarisation reference
  frame.
