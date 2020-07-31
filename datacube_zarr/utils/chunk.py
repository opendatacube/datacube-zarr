"""Functions for determining appropriate chunking for zarr storage."""

import logging
from itertools import islice, product
from typing import Dict, Hashable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import xarray as xr
from typing_extensions import Protocol

_ZARR_TARGET_CHUNK_SIZE_MB = 20.0
_DEFAULT_COMPRESSION_RATIO = 3.0

logger = logging.getLogger(__name__)


def calculate_chunk_sizes(
    sizes: Mapping[Hashable, int],
    chunk_dims: Sequence[Hashable],
    dtype: np.dtype,
    chunk_size_mb: float,
) -> Dict[Hashable, int]:
    """Determine chunk lengths on each dimension for a target chunk size (mb)."""
    fixed_dims = [d for d in sizes.keys() if d not in chunk_dims]
    fixed_size = np.prod([sizes[d] for d in fixed_dims])
    chunk_total = chunk_size_mb * (1024 ** 2) / fixed_size / dtype.itemsize
    chunks = dict(sizes)
    sorted_chunk_dims = [d for d in sorted(chunks, key=chunks.get) if d in chunk_dims]
    for i, d in enumerate(sorted_chunk_dims):
        c = int(np.round(chunk_total ** (1.0 / (len(chunk_dims) - i))))
        if chunks[d] > c:
            chunks[d] = c
        chunk_total /= chunks[d]
    return chunks


class SupportsEncode(Protocol):
    def encode(self, buf: bytes) -> bytes:
        ...


def estimate_compression_ratio(
    da: xr.DataArray,
    chunk_dims: Sequence[Hashable],
    compressor: SupportsEncode,
    test_size_mb: float = 1.0,
) -> Optional[float]:
    """Attempt to estimate the compression ratio for the data array."""
    test_chunks = calculate_chunk_sizes(da.sizes, chunk_dims, da.dtype, test_size_mb)

    def _slice_range(step: int, stop: int, max_steps: int = 10) -> List[slice]:
        """Sliding window of up to `max_step` slices each of length `step`."""
        ixs = np.arange(0, stop - (stop % step), max(step, int(stop / max_steps)))
        slices = [slice(i, i + step) for i in ixs]
        return slices

    def _eval_compression_ratio(window: Tuple[slice, ...]) -> float:
        """Compress a block of data defined by `window` and evaluate ratio."""
        block = da[dict(zip(chunk_dims, window))].data
        block_c = compressor.encode(block.tobytes())
        r = float(block.nbytes / len(block_c))
        return r

    # Randomly shuffled windows into data
    chunk_slices = [_slice_range(test_chunks[d], da.sizes[d]) for d in chunk_dims]
    chunk_windows = list(product(*chunk_slices))
    np.random.shuffle(chunk_windows)

    # Try to estimate compression ratio for "normal" data (i.e. for chunks
    # that are not all the same value)
    generate_test_ratios = (_eval_compression_ratio(w) for w in chunk_windows)
    max_tests = 15
    n_normal_req = 5
    r_normal_threshold = 10.0
    rs_normal = []
    rs_other = []
    for r in islice(generate_test_ratios, max_tests):
        if r < r_normal_threshold:
            rs_normal.append(r)
        else:
            rs_other.append(r)
        if len(rs_normal) > n_normal_req:
            break

    r_est = float(np.mean(rs_normal)) if rs_normal else None
    return r_est


def determine_dataarray_chunks(
    da: xr.DataArray,
    target_mb: float = _ZARR_TARGET_CHUNK_SIZE_MB,
    compressor: Optional[SupportsEncode] = None,
    chunk_dims: Optional[Sequence[Hashable]] = None,
    default_compression_ratio: float = _DEFAULT_COMPRESSION_RATIO,
) -> Dict[Hashable, int]:
    """Determine a good chunking for the data array based on target chunk size in MB."""

    # if data array total size is less than target then don't chunk
    if da.nbytes / (target_mb * (1024 ** 2)) < default_compression_ratio:
        chunks = dict(da.sizes)
        logger.debug(f"No chunking required for '{da.name}', {chunks}.")
        return chunks

    # default to chunking on the last 2 dimensions only
    if not chunk_dims:
        chunk_dims = da.dims[-2:]

    # determine a compression ration to use for chunk calculation
    r_est = None
    if compressor:
        r_est = estimate_compression_ratio(da, chunk_dims, compressor)
    r = r_est or default_compression_ratio

    chunks = calculate_chunk_sizes(da.sizes, chunk_dims, da.dtype, target_mb * r)
    logger.debug(
        f"Chunking '{da.name}' with {chunks} using {'evaluated' if r_est else 'default'} "
        f"compression ratio {r:.2f}."
    )
    return chunks


def auto_chunk_dataset(
    ds: xr.Dataset,
    target_mb: float = _ZARR_TARGET_CHUNK_SIZE_MB,
    compressor: Optional[SupportsEncode] = None,
    chunk_dims: Optional[Sequence[Hashable]] = None,
    default_compression_ratio: float = _DEFAULT_COMPRESSION_RATIO,
) -> xr.Dataset:
    """Chunk each array within the dataset based on target chunk size in MB."""
    for name in ds:
        chunks = determine_dataarray_chunks(
            ds[name], target_mb, compressor, chunk_dims, default_compression_ratio
        )
        ds[name] = ds[name].chunk(chunks)  # type: ignore
    return ds
