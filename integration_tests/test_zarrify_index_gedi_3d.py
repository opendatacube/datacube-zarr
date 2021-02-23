import shutil
from pathlib import Path

import pytest
import rasterio
import xarray as xr
import yaml
from click.testing import CliRunner
from datacube.api.core import Datacube
from datacube.index.hl import Doc2Dataset

from datacube_zarr.tools.set_zarr_product_extra_dim import (
    cli as set_zarr_product_extra_dim,
)
from datacube_zarr.tools.zarrify import main as zarrify
from datacube_zarr.zarr_io import replace_dataset_dim

PROJECT_ROOT = Path(__file__).parents[1]

GEDI_TEST_DATA = PROJECT_ROOT / "tests/data/gedi"
GEDI_GTIF_DATA = GEDI_TEST_DATA / "gtif"
GEDI_GTIF_PROD_DEF = GEDI_GTIF_DATA / "GEDI02_B.yaml"
GEDI_ZARR_DATA = GEDI_TEST_DATA / "zarr"
GEDI_ZARR_PROD_DEF = GEDI_ZARR_DATA / "GEDI02_B_3d_format.yaml"

GEDI_L2B_PRODUCTS = [
    "gedi_l2b",
    "gedi_l2b_cover_z",
    "gedi_l2b_pai_z",
    "gedi_l2b_pavd_z",
]


def _gedi_product_is_3d(prod):
    return prod.endswith("_z")


GEDI_RESOLUTION = (0.00027778, -0.00027778)
GEDI_CRS = "EPSG:4326"

# Bounds of the Lake Burligh Griffin dataset
LBG_LATITUDE = (-35.821784394017975, -34.988444394017975)
LBG_LONGITUDE = (148.64089852640407, 149.75201852640407)


@pytest.fixture
def gedi_zarr3d(tmp_path):
    """Zarrify GEDI rasters"""
    zarr_dir = tmp_path / "zarrs"
    runner = CliRunner()
    zarrify_args = ["--chunk", "x:512", "--chunk", "y:512", "--progress"]
    zarrify_args.extend(["--multi-dim", "--outpath", str(zarr_dir)])

    # zarrify each dataset and copy metadata
    ds = [d for d in GEDI_GTIF_DATA.iterdir() if d.is_dir()]
    for d in ds:
        dname = d.stem

        # zarrify
        res = runner.invoke(zarrify, zarrify_args + [d.as_uri()])
        assert res.exit_code == 0, res.stdout

        # copy metadata files (created by prepare script)
        meta_files = (GEDI_ZARR_DATA / dname).glob("*.yaml")
        for m in meta_files:
            shutil.copy(m, zarr_dir)

        # reset extra dims
        zarr_files = (zarr_dir / dname).glob("*.zarr")
        for z in zarr_files:
            assert z.stem.startswith(dname)
            pname = z.stem[len(dname) + 1 :]
            if _gedi_product_is_3d(pname):
                args = [
                    "--name",
                    f"gedi_l2b_{pname}_zarr",
                    z.as_uri(),
                    str(GEDI_ZARR_PROD_DEF),
                ]
                res = runner.invoke(set_zarr_product_extra_dim, args)
                assert res.exit_code == 0, res.stdout

    return zarr_dir


def test_zarrified_gedi(gedi_zarr3d):
    """Check that zarrify created a valid zarr for each input geotif."""
    ds = [d for d in GEDI_GTIF_DATA.iterdir() if d.is_dir()]
    for d in ds:
        for f in d.glob("*.tif"):
            a = xr.open_rasterio(f)
            b = xr.open_zarr(gedi_zarr3d / d.stem / f"{f.stem}.zarr")["array"].load()
            if f.stem.endswith("_z"):
                a = a.rename({"band": "z"}).assign_coords({"z": b.z.data})
            assert a.equals(b)


@pytest.fixture
def indexed_gedi_gtif(clirunner, datacube_env_name, index):
    """Add Geotiff product definition and datasets."""
    clirunner(["-v", "product", "add", str(GEDI_GTIF_PROD_DEF)])
    meta_files = [str(m) for m in GEDI_GTIF_DATA.glob("__*.yaml")]
    clirunner(["-v", "dataset", "add"] + meta_files)


@pytest.fixture
def indexed_gedi_zarr(clirunner, datacube_env_name, index, gedi_zarr3d):
    """Add Zarr product definition and datasets."""
    clirunner(["-v", "product", "add", str(GEDI_ZARR_PROD_DEF)])
    meta_files = [str(m) for m in gedi_zarr3d.glob("__*.yaml")]
    clirunner(["-v", "dataset", "add"] + meta_files)


def test_gedi_gtif_index(indexed_gedi_gtif, index):
    """Test that geotiff data is indexed."""
    dc = Datacube(index=index)
    prods = list(dc.list_products()["name"])
    for p in GEDI_L2B_PRODUCTS:
        assert p in prods

    assert len(dc.list_measurements())

    data = dc.load(
        product="gedi_l2b",
        measurements=["pai"],
        output_crs=GEDI_CRS,
        resolution=GEDI_RESOLUTION,
    )
    assert list(data.sizes.values()) == [2487, 3396, 2]


def test_gedi_zarr_index(indexed_gedi_zarr, index):
    """Test that zarr data is indexed."""
    dc = Datacube(index=index)
    prods = list(dc.list_products()["name"])
    for p in GEDI_L2B_PRODUCTS:
        assert f"{p}_zarr" in prods

    assert len(dc.list_measurements())

    data = dc.load(
        product="gedi_l2b_zarr",
        measurements=["pai"],
        output_crs=GEDI_CRS,
        resolution=GEDI_RESOLUTION,
    )
    # zarr metadata doesnt include geometry.coordinates for valid region
    assert list(data.sizes.values()) == [3011, 4195, 2]


def test_gedi_gtif_zarr_product(index, indexed_gedi_gtif, indexed_gedi_zarr):
    """Load gtif and zarr datasets and compare."""

    dc = Datacube(index=index)

    for prod in GEDI_L2B_PRODUCTS:

        # load all product measurements tiff and zarr datasets
        data_tiff = dc.load(
            product=prod,
            latitude=LBG_LATITUDE,
            longitude=LBG_LONGITUDE,
            output_crs=GEDI_CRS,
            resolution=GEDI_RESOLUTION,
        )
        data_zarr = dc.load(
            product=f"{prod}_zarr",
            latitude=LBG_LATITUDE,
            longitude=LBG_LONGITUDE,
            output_crs=GEDI_CRS,
            resolution=GEDI_RESOLUTION,
        )

        # compare datasets
        if not _gedi_product_is_3d(prod):
            # For 2D products, outputs are equal
            assert data_zarr.equals(data_tiff)

        else:
            # For 3d products all measurements indludes 2D measurements for each value in
            # the extra dimension  as well as the complete 3D measurement

            # Compare 2D measurements with gtiff product
            measurement = prod[len("gedi_l2b_") :]
            data_zarr_2d = data_zarr.drop_vars([measurement, "z"])
            assert data_zarr_2d.equals(data_tiff)

            # Compare the 3d measurement (by slicing each value)
            data_zarr_3d = data_zarr[measurement]
            for z, d_tiff in data_tiff.data_vars.items():
                d_zarr = data_zarr_3d.sel(z=int(z)).drop_vars("z").rename(z)
                assert d_zarr.equals(d_tiff)
