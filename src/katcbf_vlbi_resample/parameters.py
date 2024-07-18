# Copyright (c) 2024, National Research Foundation (SARAO)

"""Dataclasses to encapsulate configuration."""

from dataclasses import dataclass


@dataclass(frozen=True)
class StreamParameters:
    """Parameters of a time-domain stream of samples."""

    bandwidth: float  # Hz
    center_freq: float  # Hz


@dataclass(frozen=True)
class ResampleParameters:
    """Parameters controlling the resampling process."""

    fir_taps: int
    hilbert_taps: int
    passband: float
