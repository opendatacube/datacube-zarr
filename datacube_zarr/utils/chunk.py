"""Functions for determining appropriate chunking for zarr storage."""

import logging
from typing import Dict, Hashable, Mapping, Optional, Sequence, Union

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


# Defaults for calculating chunk sizes
ZARR_TARGET_CHUNK_SIZE_MB = 20.0
DEFAULT_COMPRESSION_RATIO = 2.5


def calculate_auto_chunk_sizes(
    sizes: Mapping[Hashable, int],
    chunks: Mapping[Hashable, Union[str, int]],
    chunk_total: int,
) -> Dict[Hashable, int]:
    """Determine 'auto' chunk lengths for a target chunk size.

    :param sizes: Mapping from dimension identifier to size of dimension
    :param chunks: Mapping from dimension identifier to chunk size (e.g. 'auto', -1, or N)
    :param chunk_total: Total size of chunk (i.e. number of elements)
    :return: Calculated chunk sizes of each dimension
    """
    assert set(sizes).issubset(set(chunks))

    auto_dims = [d for d in sizes.keys() if chunks[d] == "auto"]
    fixed_dims = [d for d in sizes.keys() if d not in auto_dims]
    fixed_size = np.prod(
        [sizes[d] if chunks[d] == -1 else min(chunks[d], sizes[d]) for d in fixed_dims]
    )

    if fixed_size >= chunk_total:
        auto_chunks = {d: 1 for d in auto_dims}
        logger.debug(f"Chunk size already met by fixed dims, setting: {auto_chunks}.")
    elif fixed_size * np.prod([sizes[d] for d in auto_dims]) <= chunk_total:
        auto_chunks = {d: -1 for d in auto_dims}
        logger.debug(
            f"No chunking required for 'auto' dims: { {d: sizes[d] for d in auto_dims} }."
        )
    else:
        chunk_rem = chunk_total / fixed_size
        auto_chunks = {d: sizes[d] for d in auto_dims}
        for i, d in enumerate(sorted(auto_chunks, key=auto_chunks.get)):
            c = int(np.round(chunk_rem ** (1.0 / (len(auto_dims) - i))))
            if auto_chunks[d] > c:
                auto_chunks[d] = c
            chunk_rem /= auto_chunks[d]
        logger.debug(f"Auto chunking dims: {auto_chunks}.")

    chunks_updated = {
        d: auto_chunks.get(d) or int(min(chunks[d], sizes[d])) for d in sizes
    }
    return chunks_updated


def validate_chunks(
    dims: Sequence[Hashable], chunks: Optional[Mapping[Hashable, Union[str, int]]] = None
) -> Dict[Hashable, Union[str, int]]:
    """Validate chunk dict and set default chunk size to -1 (no chunking).

    :param dims: Sequence of keys identifiying dimensions
    :param chunks: Mapping from dimension to chunk size option. Chunk size may be:
        - an integer N, for fixed chunk size
        - the integer -1, for no chunking
        - the string "auto" for automatically determined chunksize
    :return: A complete and valid chunks dict
    """
    if chunks is None:
        chunks = {}

    # Check all dimensions are valid
    invalid_dims = [d for d in chunks if d not in dims]
    if invalid_dims:
        raise ValueError(f"Invalid chunking dim(s) specified: {invalid_dims}.")

    # Check chunk values are valid
    def _is_valid_chunks(c: Union[int, str]) -> bool:
        valid = (isinstance(c, int) and (c == -1 or c > 0)) or c == "auto"
        return valid

    invalid_chunks = [(d, c) for d, c in chunks.items() if not _is_valid_chunks(c)]
    if invalid_chunks:
        raise ValueError(f"Invalid chunking value(s) specified: {invalid_chunks}.")

    # Set default to `-1` (i.e. no chunking) for dimensions not specified
    unspecified_dims = list(set(dims) - set(chunks))
    if unspecified_dims:
        logger.debug(f"Setting default 'no chunking' (-1) for dims: {unspecified_dims}")

    chunks = {d: chunks.get(d, -1) for d in dims}
    return chunks


def chunk_dataset(
    ds: xr.Dataset,
    chunks: Optional[Mapping[Hashable, Union[str, int]]] = None,
    target_mb: float = ZARR_TARGET_CHUNK_SIZE_MB,
    compression_ratio: float = DEFAULT_COMPRESSION_RATIO,
) -> xr.Dataset:
    """Chunk dataset according to chunks settings dict.

    Default behaviour if no chunks specified is to not chunk the dataset.

    :param ds: Dataset to chunk
    :param chunks: Mapping from dimension to chunk size. Chunk size may be:
        - an integer N, for fixed chunk size
        - the integer -1, for no chunking
        - the string "auto" for automatically determined chunksize
    :param target_mb: Target compressed chunk size in MB
    :param comrpression_ratio: Approx compression ratio for estimating chunk sizes
    :return: Dataset with chunked dask arrays.
    """
    chunks = validate_chunks(list(ds.dims.keys()), chunks)

    if not any(c == "auto" for c in chunks.values()):
        ds = ds.chunk(chunks)  # type: ignore
        logger.debug(f"Chunking dataset with: {chunks}")
    else:
        chunk_size_bytes = target_mb * (1024 ** 2) * compression_ratio
        for name, da in ds.data_vars.items():
            da_chunk_total = chunk_size_bytes / da.dtype.itemsize
            da_chunks = calculate_auto_chunk_sizes(da.sizes, chunks, da_chunk_total)
            logger.debug(
                f"Auto chunking array {name} with: {da_chunks} (target_mb={target_mb}, "
                f"compression_ratio={compression_ratio})"
            )
            ds[name] = da.chunk(da_chunks)  # type: ignore

    return ds
