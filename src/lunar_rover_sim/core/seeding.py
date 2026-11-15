"""Named random streams.

Every random draw in the project comes from ``named_generator(seed, stream_name)``. Each stream is seeded from the
pair (seed, CRC32 of its name), so adding a new stream never shifts the draws of an existing one, and no generator is
shared between modules.
"""
from __future__ import annotations

import zlib

import numpy as np


def stream_identifier(stream_name: str) -> int:
    """Return the 32-bit CRC of a stream name, used as the second seed word of that stream."""
    if not stream_name:
        raise ValueError("stream_name must be a non-empty string")
    return zlib.crc32(stream_name.encode("utf-8"))


def named_generator(seed: int, stream_name: str) -> np.random.Generator:
    """Return an independent numpy Generator for one named stream of one integer seed."""
    if int(seed) < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    return np.random.default_rng(np.random.SeedSequence([int(seed), stream_identifier(stream_name)]))


def derive_seed(seed: int, stream_name: str) -> int:
    """Return a deterministic 63-bit integer seed derived from (seed, stream_name), for APIs that need an integer."""
    return int(named_generator(seed, stream_name).integers(0, 2**63 - 1))
