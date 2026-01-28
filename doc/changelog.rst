Changelog
=========

.. rubric:: 0.4

This release contains **breaking changes** to the internal APIs, but the
:program:`mk_vlbi_resample` script should still work exactly the same.

- Update dependencies to newer versions.
- Allow the user of the API to specify arbitrarily labels for polarisations.
- Expose more of the polarisation parsing API as public.
- Change the way :class:`.Resample` is configured (**breaking**)
- Add a faster path when no mixing is needed (input and output centre
  frequencies are the same).
- Make it easier for downstream projects to build documentation when cupy is
  mocked out.
- Move :func:`!katcbf_vlbi_resample.mk.rechunk_seconds` to
  :meth:`katcbf_vlbi_resample.rechunk.Rechunk.align_utc_seconds`
  (**breaking**).

.. rubric:: 0.3

- Make many dependencies optional. In particular, those needed only for
  :program:`mk_vlbi_resample` but not the core functionality are placed in
  a ``cli`` extra. The installation instructions have been updated
  accordingly.

.. rubric:: 0.2

- Remove hard dependency on cupy Python package, so that users can install a
  binary wheel for it.
- Require at least cupy 13.4 to avoid a bug with non-blocking transfers.
- Change all iterators to be asynchronous, to support future integration with
  data that arrives from an asynchronous source such as the network.
- Correct documentation to indicate that Python 3.12 is the minimum supported
  version.

.. rubric:: 0.1

First public release
