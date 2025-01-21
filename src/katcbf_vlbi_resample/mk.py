# Copyright (c) 2024-2025, National Research Foundation (SARAO)

"""Main script for resampling MeerKAT HDF5 beamformer files."""

import argparse
import csv
import dataclasses
import hashlib
import json
import os
import pathlib
import warnings
from dataclasses import dataclass
from typing import Any, BinaryIO, Self

import astropy.units as u
import baseband.base.encoding
import h5py
import katsdptelstate
import numpy as np
import platformdirs
import xarray as xr
from astropy.time import Time, TimeDelta
from atomicwrites import atomic_write
from baseband.helpers.sequentialfile import FileNameSequencer
from rich.console import Console
from rich.progress import Progress

from . import cupy_bridge, hdf5_reader, polarisation, power, rechunk, resample, vdif_writer
from .parameters import ResampleParameters, StreamParameters
from .stream import ChunkwiseStream, Stream
from .utils import fraction_to_time_delta


def parse_args(threads: list[dict[str, str]]) -> argparse.Namespace:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input", nargs=2, metavar="FILENAME", help="Input single-pol HDF5 files, in the order the pols should be output"
    )
    parser.add_argument("output", metavar="PATTERN", help="Filename pattern for output files")

    input_group = parser.add_argument_group("Input options")
    input_group.add_argument("--telstate", required=True, metavar="FILENAME", help="Telescope state .rdb file")
    input_group.add_argument("--start", metavar="TIME", help="Start time in UTC [start of files]")
    input_group.add_argument("--duration", metavar="SECONDS", help="Amount of data to read [to end of files]")
    input_group.add_argument(
        "--instrument", default="narrow1", help="MK instrument whose data is captured [%(default)s]"
    )
    input_group.add_argument(
        "--time-correction",
        type=u.Quantity,
        default=u.Quantity(0.0, unit=u.s),
        help="Quantity to add to sync time to get precise sync time [%(default)s]",
    )

    output_group = parser.add_argument_group("Output options")
    output_group.add_argument("--bandwidth", type=float, required=True, metavar="HZ", help="Output bandwidth")
    output_group.add_argument("--frequency", type=float, required=True, metavar="HZ", help="Output centre frequency")
    output_group.add_argument(
        "--samples-per-frame", type=int, default=20000, metavar="SAMPLES", help="Samples per VDIF frame [%(default)s]"
    )
    output_group.add_argument("--station", default="me", metavar="ID", help="VDIF station ID [%(default)s]")
    output_group.add_argument(
        "--file-size", type=int, default=256 * 1024 * 1024, metavar="BYTES", help="Chunk file size [256 MiB]"
    )
    output_group.add_argument(
        "--record-power", type=str, metavar="FILENAME", help="CSV file in which power measurements are stored [none]"
    )

    proc_group = parser.add_argument_group("Processing options")
    proc_group.add_argument("--polarisation", type=str, metavar="A,B:C,D", help="Polarisation conversion (see manual)")
    proc_group.add_argument(
        "--fir-taps", type=int, required=True, metavar="TAPS", help="Number of taps in rational filter"
    )
    proc_group.add_argument(
        "--hilbert-taps", type=int, default=201, metavar="TAPS", help="Number of taps in Hilbert filter [%(default)s]"
    )
    proc_group.add_argument(
        "--passband",
        type=float,
        default=0.9,
        metavar="FRACTION",
        help="Fraction of band to retain in passband filter [%(default)s]",
    )
    proc_group.add_argument(
        "--threshold", type=float, default=0.969, help="Threshold (in σ) between quantisation levels [%(default)s]"
    )
    proc_group.add_argument(
        "--normalise",
        default="auto",
        help="Power normalisation method (auto, first, or 4 comma-separated values) [%(default)s]",
    )
    proc_group.add_argument("--cpu", action="store_true", help="Process on the CPU only")

    args = parser.parse_args()
    if args.start is not None:
        args.start = Time(args.start, scale="utc")
    if args.duration is not None:
        args.duration = TimeDelta(args.duration, scale="tai", format="sec")
    if args.normalise not in {"auto", "first"}:
        try:
            values = [np.float32(x) for x in args.normalise.split(",")]
            if len(values) != 4:
                raise ValueError
        except ValueError:
            parser.error("--power must be 'auto', 'first' or 4 comma-separated real values")
        args.normalise = xr.DataArray(
            np.zeros((2, 2), dtype=np.float32),
            dims=("sideband", "pol"),
            coords={
                "sideband": ["lsb", "usb"],
                "pol": ["pol0", "pol1"],
            },
        )
        for value, thread in zip(values, threads):
            args.normalise.loc[thread] = value
    if u.get_physical_type(args.time_correction) != "time":
        parser.error("--time-correction must specify units of time e.g. 2us")
    if args.polarisation is not None:
        try:
            args.polarisation = polarisation.parse_spec(args.polarisation)
        except ValueError as exc:
            parser.error(f"Invalid --polarisation: {exc}")

    return args


@dataclass
class TelescopeStateParameters:
    """Parameters inferred from katsdptelstate.

    Note: the class members must be compatible with JSON serialisation for
    caching purposes.
    """

    adc_sample_rate: float
    bandwidth: float
    center_freq: float
    sync_time: float

    def to_file(self, path: pathlib.Path) -> None:
        """Serialise to file (atomically)."""
        with atomic_write(path, overwrite=True) as fh:
            json.dump(dataclasses.asdict(self), fh)

    @classmethod
    def from_file(cls, path: os.PathLike) -> Self:
        """Deserialize a file written with :meth:`to_file`."""
        with open(path, "r") as fh:
            data = json.load(fh)
        return cls(**data)


def telescope_state_parameters(telstate: katsdptelstate.TelescopeState, instrument: str) -> TelescopeStateParameters:
    """Extract useful parameters from the telescope state."""
    stream = "antenna_channelised_voltage"
    ns = telstate.view(instrument).view(telstate.join(instrument, stream))
    return TelescopeStateParameters(
        adc_sample_rate=ns["adc_sample_rate"],
        bandwidth=ns["bandwidth"],
        center_freq=ns["center_freq"],
        sync_time=ns["sync_time"],
    )


def _ts_parameters_from_file(fh: str | os.PathLike | BinaryIO, instrument: str) -> TelescopeStateParameters:
    """Implement :func:`telescope_state_parameters_from_file` without the caching."""
    telstate = katsdptelstate.TelescopeState()
    telstate.load_from_file(fh)
    return telescope_state_parameters(telstate, instrument)


def telescope_state_parameters_from_file(filename: str | os.PathLike, instrument: str) -> TelescopeStateParameters:
    """Extract useful parameters from the telescope state.

    This uses a cache indexed by the raw content of the file.
    """
    hash_version = 1  # Update when TelescopeStateParameters changes
    params = None
    with open(filename, "rb") as fh:
        h = hashlib.file_digest(fh, "sha256")
        fh.seek(0)  # In case we need to pass it to telstate to load
        h.update(instrument.encode())
        h.update(str(hash_version).encode())
        key = h.hexdigest()
        cache_dir = pathlib.Path(platformdirs.user_cache_dir("katcbf_vlbi_resample", "SARAO"))
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = cache_dir / f"{key}.json"
            try:
                return TelescopeStateParameters.from_file(cache_file)
            except FileNotFoundError:
                pass
            params = _ts_parameters_from_file(fh, instrument)
            params.to_file(cache_file)
        except OSError as exc:
            warnings.warn(f"Could not access telescope state cache ({exc}). Loading from original.")
            if params is None:
                params = _ts_parameters_from_file(fh, instrument)
    return params


def _frac_seconds(time: Time) -> float:
    """Get number of fractional seconds since last UTC second."""
    return time.utc.ymdhms.second % 1


def rechunk_seconds(it: Stream[xr.DataArray]) -> Stream[xr.DataArray]:
    """Rechunk to align to UTC seconds.

    The alignment will not be possible if the sample rate is not an integer
    number of Hz. In this case, a warning will be printed.
    """
    # Rechunk to a chunk per UTC second. Note that this relies on having an
    # integral sampling rate.
    sample_rate = 1 / it.time_scale
    if sample_rate.denominator != 1:
        warnings.warn("Sample rate is not integral Hz, so normalisation periods will not be aligned.")
    period = round(sample_rate)

    # Fractional seconds left in the starting second
    remainder = round(-period * _frac_seconds(it.time_base)) % period
    return rechunk.Rechunk(it, round(sample_rate), remainder=remainder)


class RecordPower(power.RecordPower):
    """Record power levels to a CSV file."""

    def __init__(self, *args, threads: list[dict[str, Any]], writer, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._threads = threads
        self._writer = writer
        # Write header
        writer.writerow(["time"] + ["-".join(thread.values()) for thread in threads])

    def record_rms(self, start: int, length: int, rms: xr.DataArray) -> None:  # noqa: D102
        start_time = self.time_base + fraction_to_time_delta(start * self.time_scale)
        values = [start_time.isot]
        values.extend(rms.sel(thread).item() ** 2 for thread in self._threads)
        self._writer.writerow(values)


class ProgressStream(ChunkwiseStream[xr.DataArray, xr.DataArray]):
    """Stream that shows a progress bar on the console.

    To facilitate setting up the streams prior to putting up the progress bar,
    the progress bar is registered later by calling :meth:`set_progress`.
    """

    def __init__(self, input_data: Stream[xr.DataArray]) -> None:
        super().__init__(input_data)
        self._progress: Progress | None = None
        self._task_id = 0

    def set_progress(self, progress: Progress, task_id: int) -> None:
        """Register the progress bar that will be updated.

        The task must have a total that equals the total number of time
        samples across all chunks.
        """
        self._progress = progress
        self._task_id = task_id

    def _transform(self, chunk: xr.DataArray) -> xr.DataArray:
        if self._progress is not None:
            self._progress.update(self._task_id, advance=chunk.sizes["time"])
        return chunk


def main() -> None:  # noqa: D103
    console = Console()
    threads = [{"sideband": sideband, "pol": pol} for sideband in ["lsb", "usb"] for pol in ["pol0", "pol1"]]
    args = parse_args(threads)

    with console.status("Loading telescope state parameters..."):
        telstate_params = telescope_state_parameters_from_file(args.telstate, args.instrument)

    input_params = StreamParameters(bandwidth=telstate_params.bandwidth, center_freq=telstate_params.center_freq)
    output_params = StreamParameters(bandwidth=args.bandwidth, center_freq=args.frequency)
    resample_params = ResampleParameters(fir_taps=args.fir_taps, hilbert_taps=args.hilbert_taps, passband=args.passband)

    is_cupy = not args.cpu
    # rdcc_nbytes sets the chunk cache size. MK HDF5 files tend to have 32MB chunks
    # while the default cache size is much smaller than that, so we need to increase
    # it to actually be able to use the chunk cache.
    reader = hdf5_reader.HDF5Reader(
        {
            f"pol{i}": h5py.File(input_file, "r", rdcc_nbytes=128 * 1024 * 1024)
            for i, input_file in enumerate(args.input)
        },
        adc_sample_rate=telstate_params.adc_sample_rate,
        sync_time=Time(telstate_params.sync_time, scale="utc", format="unix") + args.time_correction,
        start_time=args.start,
        duration=args.duration,
        is_cupy=is_cupy,
    )
    n_spectra = reader.stop_spectrum - reader.start_spectrum
    progress_it = ProgressStream(reader)
    # Convert back to time domain
    it: Stream[xr.DataArray] = resample.IFFT(progress_it)
    # Get more accurate start time, now that we can do so with per-sample accuracy
    if args.start is not None:
        it = resample.ClipTime(it, start=args.start)
    # Change polarisation basis if requested
    if args.polarisation is not None:
        it = polarisation.ConvertPolarisation(it, args.polarisation)
    # Do the main resampling work
    it = resample.Resample(input_params, output_params, resample_params, it)
    # Rechunk to seconds
    it = rechunk_seconds(it)
    # Measure the power level, for both normalisation and optionally recording
    it_rms: Stream[xr.Dataset] = power.MeasurePower(it)
    if args.record_power is not None:
        # See csv module docs for explanation of newline=""
        power_fh = open(args.record_power, "w", newline="")
        it_rms = RecordPower(it_rms, threads=threads, writer=csv.writer(power_fh))
    else:
        power_fh = None
    # Normalise the power. The baseband package uses a threshold of
    # TWO_BIT_1_SIGMA so we have to adjust the level to match.
    it = power.NormalisePower(it_rms, baseband.base.encoding.TWO_BIT_1_SIGMA / args.threshold, power=args.normalise)
    # Encode to VDIF
    it = vdif_writer.VDIFEncode2Bit(it, samples_per_frame=args.samples_per_frame)
    # Transfer back to the CPU if needed
    if is_cupy:
        it = cupy_bridge.AsNumpy(it)
    frameset_it = vdif_writer.VDIFFormatter(it, threads, station=args.station, samples_per_frame=args.samples_per_frame)

    # The above just sets up an iterator. Now use it to write to file.
    fns = iter(FileNameSequencer(args.output))
    fh: baseband.vdif.VDIFFileWriter | None = None
    try:
        with Progress() as progress:
            task_id = progress.add_task("Processing...", total=n_spectra)
            progress_it.set_progress(progress, task_id)
            for frameset in frameset_it:
                if fh is None or fh.tell() + frameset.nbytes > args.file_size:
                    if fh is not None:
                        fh.close()
                    fh = baseband.vdif.open(next(fns), "wb")
                frameset.tofile(fh)
    finally:
        if fh is not None:
            fh.close()
        if power_fh is not None:
            power_fh.close()


if __name__ == "__main__":
    main()
