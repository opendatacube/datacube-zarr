import pytest
import numpy as np
import xarray as xr

from datacube_zarr.utils.chunk import (
    calculate_auto_chunk_sizes,
    chunk_dataset,
    validate_chunks,
)

DIMS = ("time", "lambda", "y", "x")
LENS = (2, 5, 10, 10)
SIZES = dict(zip(DIMS, LENS))

auto_chunk_params = [
    ([-1, -1, "auto", "auto"], 2000, [-1, -1, -1, -1]),
    ([-1, -1, "auto", "auto"], 500, [-1, -1, 7, 7]),
    ([-1, -1, "auto", "auto"], 10, [-1, -1, 1, 1]),
    ([-1, -1, -1, -1], 1, [-1, -1, -1, -1]),
    (["auto", "auto", "auto", "auto"], 100, [2, 4, 4, 3]),
    ([10, 10, "auto", "auto"], 100, [2, 5, 3, 3]),
]


@pytest.mark.parametrize("chunks_in,chunk_total,chunks_out", auto_chunk_params)
def test_calculate_auto_chunk_size(chunks_in, chunk_total, chunks_out):
    """Test auto chunking."""
    chunks = dict(zip(DIMS, chunks_in))
    chunk_auto = calculate_auto_chunk_sizes(
        sizes=dict(SIZES), chunks=chunks, chunk_total=chunk_total
    )
    assert chunk_auto == dict(zip(DIMS, chunks_out))


valid_chunk_params = [
    (None, [-1, -1, -1, -1]),
    ({}, [-1, -1, -1, -1]),
    ({"x": "auto"}, [-1, -1, -1, "auto"]),
    ({"x": 1, "y": 2, "time": 3}, [3, -1, 2, 1]),
    ({"x": 5, "something_else": 3}, [-1, -1, -1, 5]),
]


@pytest.mark.parametrize("chunks_in,chunks_out", valid_chunk_params)
def test_validate_chunks(chunks_in, chunks_out):
    """Test chunk validation."""
    validated_chunks = validate_chunks(DIMS, chunks_in)
    assert validated_chunks == dict(zip(DIMS, chunks_out))


invalid_chunk_params = [{"x": "-1"}, {"y": -3}, {"time": "Auto"}]


@pytest.mark.parametrize("chunks_in", invalid_chunk_params)
def test_invalid_chunks(chunks_in):
    """Test chunk validation."""
    with pytest.raises(ValueError):
        validate_chunks(DIMS, chunks_in)


ARRAYS = [("a", np.int8), ("b", np.float32)]
DATASET = xr.Dataset(
    {n: xr.DataArray(np.random.randn(*LENS).astype(d), dims=DIMS) for n, d in ARRAYS}
)

chunk_dataset_params = [
    (None, [2, 5, 10, 10], [2, 5, 10, 10]),
    ({"time": -1, "lambda": -1}, [2, 5, 10, 10], [2, 5, 10, 10]),
    ({"time": 1, "lambda": 1}, [1, 1, 10, 10], [1, 1, 10, 10]),
    ({"x": "auto", "y": "auto"}, [2, 5, 7, 7], [2, 5, 4, 3]),
    (
        {"time": "auto", "lambda": "auto", "x": "auto", "y": "auto"},
        [2, 5, 7, 7],
        [2, 4, 4, 4],
    ),
]


@pytest.mark.parametrize("chunks_in,chunk_a,chunk_b", chunk_dataset_params)
def test_chunk_dataset(chunks_in, chunk_a, chunk_b):
    """Test chunking dataset."""
    target_mb = 0.0005
    compression_ratio = 1.0
    ds_chunked = chunk_dataset(DATASET, chunks_in, target_mb, compression_ratio)
    assert [c[0] for c in ds_chunked["a"].chunks] == chunk_a
    assert [c[0] for c in ds_chunked["b"].chunks] == chunk_b
