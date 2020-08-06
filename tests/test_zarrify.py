"""Test zarrify cli tool."""

import json

import pytest
import click
from rasterio.crs import CRS

from datacube_zarr.tools.zarrify import ClickCRS, FileOrS3Path, KeyValue

from .utils import create_random_raster

keyvalue_params = [
    ("a:b", {}, ("a", "b")),
    ("i:8", {"value": int}, ("i", 8)),
    ("x#a,b,c", {"value": lambda v: v.split(","), "sep": "#"}, ("x", ["a", "b", "c"])),
]


def assert_expected_chunking(zarr_path, chunks, group="", name="band1"):
    metadata_path = zarr_path / '.zmetadata'
    with metadata_path.open() as fh:
        metadata = json.load(fh)
    zarray = ([group] if group else []) + [name, ".zarray"]
    actual_chunks = metadata['metadata']["/".join(zarray)]['chunks']
    assert actual_chunks == chunks


@pytest.mark.parametrize("param,kwargs,res", keyvalue_params)
def test_keyvalue_click_param(param, kwargs, res):
    """Test KeyValue click param."""
    k, v = KeyValue(**kwargs).convert(param, None, None)
    assert k == res[0]
    assert v == res[1]


def test_fileors3path_click_param(tmp_storage_path):
    """Test FileOrS3Path click param."""
    uri = tmp_storage_path.as_uri()
    path = FileOrS3Path().convert(uri, None, None)
    assert path == tmp_storage_path


def test_fileors3path_click_param_error(tmp_storage_path):
    """Test FileOrS3Path click param."""
    new_path = tmp_storage_path / "new" / "subdir"
    uri = new_path.as_uri()
    with pytest.raises(click.BadParameter):
        FileOrS3Path(exists=True).convert(uri, None, None)


clickcrs_params = [
    ("4326", CRS.from_epsg(4326)),
    (
        "+proj=longlat +ellps=GRS80 +no_defs",
        CRS.from_string("+proj=longlat +ellps=GRS80 +no_defs"),
    ),
]


@pytest.mark.parametrize("param,res", clickcrs_params)
def test_clickcrs_param(param, res):
    """Test ClickCRS click param."""
    crs = ClickCRS().convert(param, None, None)
    assert crs == res


@pytest.mark.parametrize("param", ["123433", "+proj=utm +zone=501 +south +ellps=WGS84"])
def test_clickcrs_param_error(param):
    """Test invalid ClickCRS click param."""
    with pytest.raises(click.BadParameter):
        ClickCRS().convert(param, None, None)


@pytest.mark.parametrize("options", ([], ["--outpath", "./", "--inplace"]))
def test_zarrify_bad_inplace_options(zarrifycli, tmp_path, options):
    """Test bad inplace/outpath options."""
    res = zarrifycli(options + [str(tmp_path)])
    assert res.exit_code != 0, res.stdout


bad_chunks = [
    "--chunk x:none",
    "--chunk x:2.2",
    "--auto-chunk --chunk x:100",
    "--chunk z:100",
]


@pytest.mark.parametrize("chunks", bad_chunks)
def test_zarrify_bad_chunk_options(zarrifycli, tmp_path, chunks):
    """Test bad inplace/outpath options."""
    raster = create_random_raster(tmp_path)
    res = zarrifycli([str(raster), "--inplace"] + chunks.split())
    assert res.exit_code != 0, res.stdout


chunk_params = [
    ([], [200, 300]),
    (["--chunk", "x:50", "--chunk", "y:30"], [30, 50]),
    (["--auto-chunk", "--chunk-target-mb", "0.01"], [81, 81]),
    (["--chunk", "x:100", "--chunk", "y:auto", "--chunk-target-mb", "0.01"], [66, 100]),
    (["--chunk", "x:-1", "--chunk", "y:auto", "--chunk-target-mb", "0.01"], [22, 300]),
    (["--chunk", "y:auto"], [200, 300]),
    (["--chunk-target-mb", "0.0001", "--chunk", "y:auto"], [1, 300]),
]


@pytest.mark.parametrize("chunks_opts,chunks", chunk_params)
def test_zarrify(zarrifycli, tmp_raster, chunks_opts, chunks):
    """Test zarrify cli."""
    res = zarrifycli(["--inplace", tmp_raster.as_uri()] + chunks_opts)
    assert res.exit_code == 0, res.stdout
    assert not tmp_raster.exists()
    zarr_path = tmp_raster.parent / f"{tmp_raster.stem}.zarr"
    assert zarr_path.exists()
    assert_expected_chunking(zarr_path, chunks)


def test_zarrify_no_dataset(zarrifycli, tmp_path):
    """Test no dataset."""
    res = zarrifycli(["--inplace", str(tmp_path / "dataset")])
    assert res.exit_code != 0, res.stdout


def test_zarrify_unsuppported_dataset(zarrifycli, tmp_path):
    """Test no dataset."""
    ds = tmp_path / "dataset.xyz"
    ds.touch()
    res = zarrifycli(["--inplace", str(ds)])
    assert res.exit_code != 0, res.stdout


def test_zarrify_ignore_dataset(zarrifycli, tmp_raster):
    """Test no dataset."""
    res = zarrifycli(["--inplace", "--ignore", "*.tif", tmp_raster.as_uri()])
    assert res.exit_code == 0, res.stdout
    assert tmp_raster.exists()
    assert not (tmp_raster.parent / f"{tmp_raster.stem}.zarr").exists()
