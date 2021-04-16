from pathlib import Path

import pytest
import xarray as xr
import yaml
from click.testing import CliRunner
from datacube.api.core import Datacube

from datacube_zarr.tools.set_zarr_product_extra_dim import (
    cli as set_zarr_product_extra_dim,
)
from datacube_zarr.tools.zarrify import cli as zarrify

PROJECT_ROOT = Path(__file__).parents[1]

GEDI_TEST_DATA = PROJECT_ROOT / "tests/data/gedi"
GEDI_GTIF_DATA = GEDI_TEST_DATA / "gtif"
GEDI_GTIF_PROD_DEF = GEDI_GTIF_DATA / "GEDI02_B.yaml"
GEDI_ZARR_DATA = GEDI_TEST_DATA / "zarr"
GEDI_ZARR_PROD_DEF = GEDI_ZARR_DATA / "GEDI02_B_3d_format.yaml"

GEDI_L2B_TEST_PRODUCTS = {
    "gedi_l2b": ["beam", "pai"],
    "gedi_l2b_cover_z": None,
}


def _gedi_product_is_3d(prod):
    return prod.endswith("_z")


def custom_dumb_fuser(dst, src):
    dst[:] = src[:]


GEDI_RESOLUTION = (0.00027778, -0.00027778)
GEDI_CRS = "EPSG:4326"

# Bounds of the Lake Burligh Griffin dataset
LBG_LATITUDE = (-35.821784394017975, -34.988444394017975)
LBG_LONGITUDE = (148.64089852640407, 149.75201852640407)


def _copy_gedi_metadata(indir, outdir, merged=False):
    """copy metadata files (created by prepare script)."""

    def _update_merged_measurement_path(meas):
        base, parent, name = meas["path"].rsplit("/", 2)
        z = f"{parent}_.zarr#{name.split('.zarr')[0]}"
        meas["path"] = "/".join([base, parent, z])
        return meas

    meta_files = indir.glob("__*.yaml")
    meta_files_out = []
    for m in meta_files:
        meta = yaml.safe_load(m.read_text())
        if merged:
            meta["measurements"] = {
                k: _update_merged_measurement_path(v)
                for k, v in meta["measurements"].items()
            }

        mout = outdir / m.name
        mout.write_text(yaml.dump(meta))
        meta_files_out.append(mout)

    return meta_files_out


def zarrify_gedi_data(outdir, merged=False):
    """Zarrify GEDI rasters"""
    runner = CliRunner()
    zarrify_args = ["--chunk", "x:512", "--chunk", "y:512", "--progress"]
    zarrify_args.extend(["--outpath", str(outdir)])
    if merged:
        zarrify_args.extend(["--merge-datasets-per-dir"])

    # zarrify each dataset and copy metadata
    ds = [d for d in GEDI_GTIF_DATA.iterdir() if d.is_dir()]
    for d in ds:
        dname = d.stem

        # zarrify
        res = runner.invoke(zarrify, zarrify_args + [d.as_uri()])
        assert res.exit_code == 0, res.stdout

        # copy metadata files (created by prepare script)
        mfiles = _copy_gedi_metadata(GEDI_ZARR_DATA / dname, outdir, merged=merged)

        # reset extra dims
        for m in mfiles:
            assert dname in m.stem
            pname = m.stem[len(dname) + 3 :]
            if _gedi_product_is_3d(pname):
                meta = yaml.safe_load(m.read_text())
                zurl = "/".join(
                    ["file:/", str(m.parent), meta["measurements"].popitem()[1]["path"]]
                )
                args = [
                    "--name",
                    f"{pname}_zarr",
                    zurl,
                    str(GEDI_ZARR_PROD_DEF),
                ]
                res = runner.invoke(set_zarr_product_extra_dim, args)
                assert res.exit_code == 0, res.stdout


@pytest.fixture(scope="session")
def gedi_zarr3d(tmp_path_factory):
    """Zarrify rasters individually."""
    zarr_dir = tmp_path_factory.mktemp("gedi_3d_zarrs")
    zarrify_gedi_data(zarr_dir)
    return zarr_dir


@pytest.fixture(scope="session")
def gedi_zarr3d_merged(tmp_path_factory):
    """Zarrify rasters into a single zarr."""
    zarr_dir = tmp_path_factory.mktemp("gedi_3d_zarrs_merged")
    zarrify_gedi_data(zarr_dir, merged=True)
    return zarr_dir


def test_zarrified_gedi(gedi_zarr3d):
    """Check that zarrify created a valid zarr for each input geotif."""
    ds = [d for d in GEDI_GTIF_DATA.iterdir() if d.is_dir()]
    for d in ds:
        for f in d.glob("*.tif"):
            a = xr.open_rasterio(f)
            if a.shape[0] == 1:
                a = a.sel(band=1, drop=True)
            b = xr.open_zarr(gedi_zarr3d / d.stem / f"{f.stem}.zarr")["array"].load()
            if f.stem.endswith("_z"):
                a = a.rename({"band": "z"}).assign_coords({"z": b.z.data})
            assert a.equals(b)


def test_zarrified_gedi_merged(gedi_zarr3d_merged):
    """Check that zarrify created a valid zarr for each input dir."""
    ds = [d for d in GEDI_GTIF_DATA.iterdir() if d.is_dir()]
    for d in ds:
        zpath = gedi_zarr3d_merged / d.stem / f"{d.stem}_.zarr"
        assert zpath.exists()
        for f in d.glob("*.tif"):
            a = xr.open_rasterio(f)
            if a.shape[0] == 1:
                a = a.sel(band=1, drop=True)
            b = xr.open_zarr(zpath, group=f.stem)["array"].load()
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


@pytest.fixture
def indexed_gedi_zarr_merged(clirunner, datacube_env_name, index, gedi_zarr3d_merged):
    """Add merged Zarr product definition and datasets."""
    clirunner(["-v", "product", "add", str(GEDI_ZARR_PROD_DEF)])
    meta_files = [str(m) for m in gedi_zarr3d_merged.glob("__*.yaml")]
    res = clirunner(["-v", "dataset", "add"] + meta_files)
    assert res.exit_code == 0


def test_gedi_gtif_index(indexed_gedi_gtif, index):
    """Test that geotiff data is indexed."""
    dc = Datacube(index=index)
    prods = list(dc.list_products()["name"])
    for p in GEDI_L2B_TEST_PRODUCTS:
        assert p in prods

    assert len(dc.list_measurements())

    data = dc.load(
        product="gedi_l2b",
        measurements=["pai"],
        output_crs=GEDI_CRS,
        resolution=GEDI_RESOLUTION,
    )
    assert list(data.sizes.values()) == [425, 557, 2]
    assert not (data['pai'].values == data['pai'].nodata).all()


def test_gedi_zarr_index(indexed_gedi_zarr, index):
    """Test that zarr data is indexed."""
    dc = Datacube(index=index)
    prods = list(dc.list_products()["name"])
    for p in GEDI_L2B_TEST_PRODUCTS:
        assert f"{p}_zarr" in prods

    assert len(dc.list_measurements())

    data = dc.load(
        product="gedi_l2b_zarr",
        measurements=["pai"],
        output_crs=GEDI_CRS,
        resolution=GEDI_RESOLUTION,
    )
    # zarr metadata doesnt include geometry.coordinates for valid region
    assert list(data.sizes.values()) == [421, 552, 2]
    assert not (data['pai'].values == data['pai'].nodata).all()


def test_gedi_zarr_index_merged(indexed_gedi_zarr_merged, index):
    """Test that zarr data is indexed."""
    dc = Datacube(index=index)
    prods = list(dc.list_products()["name"])
    for p in GEDI_L2B_TEST_PRODUCTS:
        assert f"{p}_zarr" in prods

    assert len(dc.list_measurements())

    data = dc.load(
        product="gedi_l2b_zarr",
        measurements=["pai"],
        output_crs=GEDI_CRS,
        resolution=GEDI_RESOLUTION,
    )
    # zarr metadata doesnt include geometry.coordinates for valid region
    assert list(data.sizes.values()) == [421, 552, 2]
    assert not (data['pai'].values == data['pai'].nodata).all()


def stack_3d_on_z(ds, name):
    """Stack 'z' measurments together along new 3D 'z' dimension."""
    da = ds.to_array(dim="z")
    da = da.assign_coords(z=da.z.astype(float))
    dim_order = ['time', 'z', *da.dims[-2:]]
    da = da.transpose(*dim_order)
    ds3 = da.to_dataset(name=name)
    return ds3


def _compare_all_gedi_gtif_zarr_products(index):
    """Load gtif and zarr datasets and compare."""

    dc = Datacube(index=index)

    for prod, meas in GEDI_L2B_TEST_PRODUCTS.items():

        # load all product measurements tiff and zarr datasets
        data_tiff = dc.load(
            product=prod,
            measurements=meas,
            latitude=LBG_LATITUDE,
            longitude=LBG_LONGITUDE,
            output_crs=GEDI_CRS,
            resolution=GEDI_RESOLUTION,
        )
        data_zarr = dc.load(
            product=f"{prod}_zarr",
            measurements=meas,
            latitude=LBG_LATITUDE,
            longitude=LBG_LONGITUDE,
            output_crs=GEDI_CRS,
            resolution=GEDI_RESOLUTION,
        )
        for da in data_zarr.data_vars.values():
            assert da.size

        # compare datasets
        if not _gedi_product_is_3d(prod):
            # For 2D products, outputs are equal
            xr.testing.assert_equal(data_zarr, data_tiff)

        else:
            # For 3d products all measurements indludes 2D measurements for each value in
            # the extra dimension  as well as the complete 3D measurement

            # Compare 2D measurements with gtiff product
            measurement = prod[len("gedi_l2b_") :]
            data_zarr_2d = data_zarr.drop_vars([measurement, "z"])
            xr.testing.assert_equal(data_zarr_2d, data_tiff)

            # Merge the tiff 2d measurement to 3d and compare to zarr
            data_zarr_3d = data_zarr[measurement]
            data_tiff_3d = stack_3d_on_z(data_tiff, measurement)[measurement]
            xr.testing.assert_equal(data_tiff_3d, data_zarr_3d)


def test_gedi_gtif_zarr_product(index, indexed_gedi_gtif, indexed_gedi_zarr):
    """Test loaded zarr/gtif products match."""
    _compare_all_gedi_gtif_zarr_products(index)


def test_gedi_gtif_zarr_product_merged(
    index, indexed_gedi_gtif, indexed_gedi_zarr_merged
):
    """Test loaded zarr(merged)/gtif products match."""
    _compare_all_gedi_gtif_zarr_products(index)


@pytest.mark.parametrize("z_query", [5, (5, 15), 50, (50, 75)])
def test_gedi_load_extradim_slice(
    index, indexed_gedi_gtif, indexed_gedi_zarr_merged, z_query
):
    dc = Datacube(index=index)
    prod = "gedi_l2b_cover_z"

    if isinstance(z_query, int):
        meas = [str(z_query)]
    else:
        meas = [str(z) for z in range(z_query[0], z_query[1] + 5, 5)]

    data_tiff = dc.load(
        product=prod,
        measurements=meas,
        latitude=LBG_LATITUDE,
        longitude=LBG_LONGITUDE,
        output_crs=GEDI_CRS,
        resolution=GEDI_RESOLUTION,
    )
    data_tiff_3d = stack_3d_on_z(data_tiff, "cover_z")
    del data_tiff
    data_zarr = dc.load(
        product=f"{prod}_zarr",
        measurements=["cover_z"],
        latitude=LBG_LATITUDE,
        longitude=LBG_LONGITUDE,
        output_crs=GEDI_CRS,
        resolution=GEDI_RESOLUTION,
        z=z_query,
    )
    xr.testing.assert_equal(data_zarr, data_tiff_3d)


def test_3d_reprojection(index, indexed_gedi_gtif, indexed_gedi_zarr):
    """Reproject to GDA94 (MGA Zone 55)."""
    dc = Datacube(index=index)
    prod = "gedi_l2b_cover_z"
    measurement = "cover_z"
    output_crs = "EPSG:28355"
    resolution = (100, -100)
    resampling = "cubic"

    # Slice on z dimension
    z_range = (20, 30)
    z_meas = ["20", "25", "30"]

    data_tiff = dc.load(
        product=prod,
        measurements=z_meas,
        latitude=LBG_LATITUDE,
        longitude=LBG_LONGITUDE,
        output_crs=output_crs,
        resolution=resolution,
        resampling=resampling,
    )
    data_tiff_3d = stack_3d_on_z(data_tiff, measurement)
    del data_tiff
    data_zarr = dc.load(
        product=f"{prod}_zarr",
        measurements=[measurement],
        latitude=LBG_LATITUDE,
        longitude=LBG_LONGITUDE,
        output_crs=output_crs,
        resolution=resolution,
        resampling=resampling,
        z=z_range,
    )
    assert data_zarr[measurement].size
    xr.testing.assert_equal(data_zarr, data_tiff_3d)


def test_3d_dask_chunks(index, indexed_gedi_gtif, indexed_gedi_zarr):
    """Test dask load."""

    dc = Datacube(index=index)
    prod = "gedi_l2b_cover_z"
    measurement = "cover_z"

    data_tiff = dc.load(
        product=prod,
        latitude=LBG_LATITUDE,
        longitude=LBG_LONGITUDE,
        output_crs=GEDI_CRS,
        resolution=GEDI_RESOLUTION,
        dask_chunks={'time': 2},
    )
    data_tiff_3d = stack_3d_on_z(data_tiff, measurement)
    del data_tiff
    data_zarr = dc.load(
        product=f"{prod}_zarr",
        measurements=[measurement],
        latitude=LBG_LATITUDE,
        longitude=LBG_LONGITUDE,
        output_crs=GEDI_CRS,
        resolution=GEDI_RESOLUTION,
        dask_chunks={'time': 2, 'z': 15},
    )
    assert data_zarr[measurement].size
    xr.testing.assert_equal(data_zarr, data_tiff_3d)
