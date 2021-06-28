'''Helper functions for the unit tests.'''
import json
import re
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import rasterio
import xarray as xr
from rasterio.crs import CRS

from datacube_zarr import ZarrIO


def create_random_raster_local(
    outdir: Path,
    label: str = "raster",
    height: int = 200,
    width: int = 300,
    nbands: int = 1,
) -> Path:
    """Create a raster with random data."""
    outdir.mkdir(parents=True, exist_ok=True)
    dtype = np.float32
    data = np.random.randn(nbands, height, width).astype(dtype)
    file_path = outdir / f"{label}_{nbands}x{height}x{width}.tif"

    bbox = [149, 35, 150, 36]
    transform = rasterio.transform.from_bounds(*bbox, width, height)
    meta = {
        "driver": "GTiff",
        "count": nbands,
        "width": width,
        "height": height,
        "crs": CRS.from_epsg("4326"),
        "nodata": None,
        "dtype": dtype,
        "transform": transform,
    }

    with rasterio.open(file_path.as_uri(), "w", **meta) as dst:
        dst.write(data)

    return file_path


def create_random_raster(outdir: Path, **kwargs: Any) -> Path:
    """Create random raster on s3 or locally."""
    if outdir.as_uri().startswith("file://"):
        raster_file = create_random_raster_local(outdir, **kwargs)
    else:
        with TemporaryDirectory() as savedir:
            tmp_file = create_random_raster_local(Path(savedir), **kwargs)
            raster_file = outdir / tmp_file.relative_to(Path(savedir))
            raster_file.write_bytes(tmp_file.read_bytes())
    return raster_file


def copytree(p1: Path, p2: Path) -> None:
    """Copytree for local/s3 paths."""
    for o1 in p1.iterdir():
        o2 = p2 / o1.name
        if o1.is_dir():
            copytree(o1, o2)
        else:
            if o2.as_uri().startswith("file") and not o2.parent.exists():
                o2.parent.mkdir(parents=True)
            o2.write_bytes(o1.read_bytes())


def _load_dataset(uri):
    '''Load Dataset from storage.'''
    zio = ZarrIO()
    return zio.open_dataset(uri=uri).load()


def _save_dataset(data, uri, name, chunks=None):
    '''Save Dataset to storage.'''
    zio = ZarrIO()
    zio.save_dataset(uri=uri, dataset=data.copy().to_dataset(name=name), chunks=chunks)


def raster_and_zarr_are_equal(raster_uri, zarr_uri, separate_bands=False):
    """Compare raster and zarr files."""
    da_raster = xr.open_rasterio(raster_uri)
    if da_raster.shape[0] == 1:
        da_raster = da_raster.sel(band=1, drop=True)

    ds_zarr = _load_dataset(zarr_uri)

    if separate_bands is False:
        da_zarr = ds_zarr["array"]
    else:
        sorted_vars = sorted(ds_zarr.data_vars.values(), key=lambda v: int(v.name[4:]))
        da_zarr = xr.concat(sorted_vars, dim="band").assign_coords(
            {"band": list(range(1, len(ds_zarr) + 1))}
        )
    data_coords_dims_equal = da_raster.equals(da_zarr)
    crs_equal = CRS.from_string(da_raster.crs) == CRS.from_string(da_zarr.crs)
    return data_coords_dims_equal and crs_equal


def _check_zarr_files(data, root, group, name, chunks):
    """Check zarr objects in local/s3 filesystem.

    Only some metadata and chunk file names are checked, not actual binary content."""
    assert root.exists(), f'Missing {root} after save'

    # Check chunks in root level metadata
    zmeta = root / '.zmetadata'
    assert zmeta.exists(), "Missing consolidated metadata"
    metadata = json.loads(zmeta.read_text())
    metadata_chunks = metadata['metadata'][f'{group}/{name}/.zarray']['chunks']
    assert metadata_chunks == chunks['output'], 'Chunks not as set'

    dataset_dir = root / group
    assert dataset_dir.exists(), f'Missing {group}/ in {root}'
    array_dir = dataset_dir / name
    assert array_dir.exists(), f'Missing {name}/ in {dataset_dir}'

    # Check chunks in array level metadata
    zarray = array_dir / '.zarray'
    assert zarray.exists(), f'Missing .zarray in {array_dir}'
    metadata = json.loads(zarray.read_text())
    assert metadata['chunks'] == chunks['output'], 'Chunks not as set'
    assert metadata['shape'] == list(data.shape), 'Data shape not as set'

    # Check chunk files
    chunk_files = sorted([path.name for path in array_dir.glob('?.?')])
    expected_chunk_files = sorted(
        [
            f'{i}.{j}'
            for i in range(chunks['chunks_per_side'])
            for j in range(chunks['chunks_per_side'])
        ]
    )
    assert chunk_files == expected_chunk_files, 'Unexpected chunk files'


def message_is_logged(log_record, text, logger=None, level=None):
    """Check if `text` is present in log record."""

    def _match(msg_logger, msg_level, msg_text):
        return (
            (logger is None or logger == msg_logger)
            and (level is None or level == msg_level)
            and re.match(text, msg_text)
        )

    return any(_match(*rec) for rec in log_record.record_tuples)
