from pathlib import Path

import pytest
import rasterio
import yaml
from click.testing import CliRunner
from datacube.api.core import Datacube
from datacube.index.hl import Doc2Dataset

from examples.prepare_zarr_ls5 import main as prepare_zarr_ls5
from tools.zarrify import main as zarrify

PROJECT_ROOT = Path(__file__).parents[1]
CONFIG_SAMPLES = PROJECT_ROOT / "docs/config_samples/"
LS5_DATASET_TYPES = CONFIG_SAMPLES / "dataset_types/ls5_scenes.yaml"
LS5_DATASET_TYPES_ZARR = CONFIG_SAMPLES / "dataset_types/ls5_scenes_zarr.yaml"
TEST_DATA = PROJECT_ROOT / "tests" / "data" / "lbg"
LBG_NBAR = "LS5_TM_NBAR_P54_GANBAR01-002_090_084_19920323"
LBG_PQ = "LS5_TM_PQ_P55_GAPQ01-002_090_084_19920323"


def test_ls5_dataset_access(ls5_dataset_path):
    """Test rasterio can access ls5 data."""
    raster = (
        ls5_dataset_path
        / "LS5_TM_NBAR_P54_GANBAR01-002_090_084_19920323"
        / "scene01"
        / "LS5_TM_NBAR_P54_GANBAR01-002_090_084_19920323_B10.tif"
    )
    rasterio.open(raster.as_uri()).close()


@pytest.mark.usefixtures("default_metadata_type")
@pytest.mark.parametrize("datacube_env_name", ("datacube",))
@pytest.mark.parametrize("convert_inplace", (False, True))
def test_zarrify_prepare_index(
    clirunner, tmp_path, convert_inplace, datacube_env_name, index, ls5_dataset_path
):
    """Convert test ls5 tifs to zarr, prepare mdetadata, index and compare."""

    # Add the geotiff LS5 product and dataset
    clirunner(["-v", "product", "add", str(LS5_DATASET_TYPES)])
    clirunner(["-v", "dataset", "add", str(TEST_DATA / LBG_NBAR)])
    clirunner(["-v", "dataset", "add", str(TEST_DATA / LBG_PQ)])

    # zarrify geotiffs
    runner = CliRunner()
    zarrify_args = ["--chunk", "x:500", "--chunk", "y:500"]
    if convert_inplace:
        zarrify_args.append("--inplace")
        zarr_dir = ls5_dataset_path.parent
    else:
        zarr_dir = tmp_path / "zarrs"
        zarrify_args.extend(["--outpath", str(zarr_dir)])

    zarrify_args.append(ls5_dataset_path.as_uri())
    res_zarrify = runner.invoke(zarrify, zarrify_args)
    assert res_zarrify.exit_code == 0, res_zarrify.stdout

    zarr_dataset_dir = zarr_dir / "lbg"

    # prepare metadata for zarr
    res_prep = runner.invoke(prepare_zarr_ls5, [(zarr_dataset_dir / LBG_NBAR).as_uri()])
    assert res_prep.exit_code == 0, res_prep.stdout

    # Add the zarr LS5 products
    clirunner(["-v", "product", "add", str(LS5_DATASET_TYPES_ZARR)])

    # Add the zarr datasets
    for ds in (LBG_NBAR, LBG_PQ):
        ds_dir = zarr_dataset_dir / ds
        if ds_dir.as_uri().startswith("s3"):
            # Quick index from s3. Recommended process:
            # `s3-find --skip-check '<base_uri>/*/*yaml' | \
            #   s3-to-tar | dc-index-from-tar --env <env> --ignore-lineage`
            doc2ds = Doc2Dataset(index, skip_lineage=True)
            zarr_meta_s3 = ds_dir / "agdc-metadata.yaml"
            doc = yaml.safe_load(zarr_meta_s3.read_text())
            ds, err = doc2ds(doc, zarr_meta_s3.as_uri())
            assert ds is not None, err
            index.datasets.add(ds)
        else:
            clirunner(["-v", "dataset", "add", str(ds_dir)])

    # LS5 NBAR scene params
    output_crs = "EPSG:28355"
    resolution = (25, -25)
    latitude = (-35.282468, -35.315409)
    longitude = (149.0689, 149.156705)

    # Load data
    dc = Datacube(index=index)

    for prod in ("ls5_nbar_scene",):
        data_tiff = dc.load(
            product=prod,
            latitude=latitude,
            longitude=longitude,
            output_crs=output_crs,
            resolution=resolution,
        )
        data_zarr = dc.load(
            product=f"{prod}_zarr",
            latitude=latitude,
            longitude=longitude,
            output_crs=output_crs,
            resolution=resolution,
        )

    # compare datasets
    assert data_zarr.equals(data_tiff)
