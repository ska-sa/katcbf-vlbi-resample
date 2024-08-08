# Copyright (c) 2024, National Research Foundation (SARAO)

"""Main script for resampling MeerKAT HDF5 beamformer files."""

import argparse
from dataclasses import dataclass

import baseband.base.encoding
import h5py
import katsdptelstate
import xarray as xr
from astropy.time import Time, TimeDelta
from baseband.helpers.sequentialfile import FileNameSequencer

from . import cupy_bridge, hdf5_reader, power, resample, vdif_writer
from .parameters import ResampleParameters, StreamParameters
from .stream import Stream


def parse_args() -> argparse.Namespace:
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

    output_group = parser.add_argument_group("Output options")
    output_group.add_argument("--bandwidth", type=float, required=True, metavar="HZ", help="Output bandwidth")
    output_group.add_argument("--frequency", type=float, required=True, metavar="HZ", help="Output centre frequency")
    output_group.add_argument(
        "--samples-per-frame", type=int, default=1000000, metavar="SAMPLES", help="Samples per VDIF frame [%(default)s]"
    )
    output_group.add_argument("--station", default="me", metavar="ID", help="VDIF station ID [%(default)s]")
    output_group.add_argument(
        "--file-size", type=int, default=256 * 1024 * 1024, metavar="BYTES", help="Chunk file size [256 MiB]"
    )

    proc_group = parser.add_argument_group("Processing options")
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
    proc_group.add_argument("--cpu", action="store_true", help="Process on the CPU only")

    args = parser.parse_args()
    if args.start is not None:
        args.start = Time(args.start, scale="utc")
    if args.duration is not None:
        args.duration = TimeDelta(args.duration, scale="tai", format="sec")

    return args


@dataclass
class TelescopeStateParameters:
    """Parameters inferred from katsdptelstate."""

    adc_sample_rate: float
    bandwidth: float
    center_freq: float
    sync_time: Time


def telescope_state_parameters(telstate: katsdptelstate.TelescopeState, instrument: str) -> TelescopeStateParameters:
    """Extract useful parameters from the telescope state."""
    stream = "antenna_channelised_voltage"
    ns = telstate.view(instrument).view(telstate.join(instrument, stream))
    return TelescopeStateParameters(
        adc_sample_rate=ns["adc_sample_rate"],
        bandwidth=ns["bandwidth"],
        center_freq=ns["center_freq"],
        sync_time=Time(ns["sync_time"], scale="utc", format="unix"),
    )


def main() -> None:  # noqa: D103
    args = parse_args()

    telstate = katsdptelstate.TelescopeState()
    telstate.load_from_file(args.telstate)
    telstate_params = telescope_state_parameters(telstate, args.instrument)

    input_params = StreamParameters(bandwidth=telstate_params.bandwidth, center_freq=telstate_params.center_freq)
    output_params = StreamParameters(bandwidth=args.bandwidth, center_freq=args.frequency)
    resample_params = ResampleParameters(fir_taps=args.fir_taps, hilbert_taps=args.hilbert_taps, passband=args.passband)

    is_cupy = not args.cpu
    # rdcc_nbytes sets the chunk cache size. MK HDF5 files tend to have 32MB chunks
    # while the default cache size is much smaller than that, so we need to increase
    # it to actually be able to use the chunk cache.
    it: Stream[xr.DataArray] = hdf5_reader.HDF5Reader(
        {
            f"pol{i}": h5py.File(input_file, "r", rdcc_nbytes=128 * 1024 * 1024)
            for i, input_file in enumerate(args.input)
        },
        adc_sample_rate=telstate_params.adc_sample_rate,
        sync_time=telstate_params.sync_time,
        start_time=args.start,
        duration=args.duration,
        is_cupy=is_cupy,
    )
    # Convert back to time domain
    it = resample.IFFT(it)
    # Get more accurate start time, now that we can do so with per-sample accuracy
    if args.start is not None:
        it = resample.ClipTime(it, start=args.start)
    # Do the main resampling work
    it = resample.Resample(input_params, output_params, resample_params, it)
    # Normalise the power. The baseband package uses a threshold of
    # TWO_BIT_1_SIGMA so we have to adjust the level to match.
    it = power.NormalisePower(it, baseband.base.encoding.TWO_BIT_1_SIGMA / args.threshold)
    # Encode to VDIF
    it = vdif_writer.VDIFEncode2Bit(it, samples_per_frame=args.samples_per_frame)
    # Transfer back to the CPU if needed
    if is_cupy:
        it = cupy_bridge.AsNumpy(it)
    threads = [{"sideband": sideband, "pol": pol} for sideband in ["lsb", "usb"] for pol in ["pol0", "pol1"]]
    frameset_it = vdif_writer.VDIFFormatter(it, threads, station=args.station, samples_per_frame=args.samples_per_frame)

    # The above just sets up an iterator. Now use it to write to file.
    print("Ready to start")
    fns = iter(FileNameSequencer(args.output))
    fh: baseband.vdif.VDIFFileWriter | None = None
    try:
        for frameset in frameset_it:
            if fh is None or fh.tell() + frameset.nbytes > args.file_size:
                if fh is not None:
                    fh.close()
                fh = baseband.vdif.open(next(fns), "wb")
            frameset.tofile(fh)
    finally:
        if fh is not None:
            fh.close()


if __name__ == "__main__":
    main()
