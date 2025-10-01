Changelog
=========

.. rubric:: 0.2.0

- Remove hard dependency on cupy Python package, so that users can install a
  binary wheel for it.
- Require at least cupy 13.4 to avoid a bug with non-blocking transfers.
- Change all iterators to be asynchronous, to support future integration with
  data that arrives from an asynchronous source such as the network.
- Correct documentation to indicate that Python 3.12 is the minimum supported
  version.

.. rubric:: 0.1.0

First public release
