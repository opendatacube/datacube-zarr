import shutil
from pathlib import Path

import pytest
from click.testing import CliRunner

#import datacube
from datacube.api.core import Datacube
from examples.prepare_zarr_ls5 import main as prepare_zarr_ls5
from tools.zarrify import main as zarrify


PROJECT_ROOT = Path(__file__).parents[1]
CONFIG_SAMPLES = PROJECT_ROOT / "docs/config_samples/"
LS5_DATASET_TYPES = CONFIG_SAMPLES / "dataset_types/ls5_scenes.yaml"
LS5_DATASET_TYPES_ZARR = CONFIG_SAMPLES / "dataset_types/ls5_scenes_zarr.yaml"
TEST_DATA = PROJECT_ROOT / "tests" / "data" / "lbg"
LBG_NBAR = "LS5_TM_NBAR_P54_GANBAR01-002_090_084_19920323"


@pytest.mark.usefixtures("default_metadata_type")
@pytest.mark.parametrize("datacube_env_name", ("datacube",))
@pytest.mark.parametrize("convert_inplace", (False, True))
def test_zarrify_prepare_index(
    clirunner, tmpdir, convert_inplace, datacube_env_name, index,
):
    """Convert test ls5 tifs to zarr, prepare mdetadata, index and compare."""
    lbg_dir = Path(tmpdir) / "geotifs" / "lbg"
    zarr_dir = Path(tmpdir) / "zarrs"

    # copy test data to tmp dir
    shutil.copytree(str(TEST_DATA), str(lbg_dir))

    # Add the geotiff LS5 product and dataset
    clirunner(["-v", "product", "add", str(LS5_DATASET_TYPES)])
    clirunner(["-v", "dataset", "add", str(lbg_dir / LBG_NBAR)])

    # zarrify geotiffs
    runner = CliRunner()
    zarrify_args = ["--chunk", "x:500", "--chunk", "y:500"]
    if convert_inplace:
        zarrify_args.append("--inplace")
        dataset_path = zarr_dir / "lbg"
        shutil.copytree(str(lbg_dir), str(dataset_path))
    else:
        zarrify_args.extend(["--outpath", str(zarr_dir)])
        dataset_path = lbg_dir

    zarrify_args.append(str(dataset_path))
    runner.invoke(zarrify, zarrify_args)

    zarr_dataset_dir = zarr_dir / "lbg" / LBG_NBAR

    # prepare metadata for zarr
    runner.invoke(prepare_zarr_ls5, [str(zarr_dataset_dir)])

    # Add the zarr LS5 product and dataset
    clirunner(["-v", "product", "add", str(LS5_DATASET_TYPES_ZARR)])
    clirunner(["-v", "dataset", "add", str(zarr_dataset_dir)])

    # LS5 NBAR scene params
    output_crs = "EPSG:28355"
    resolution = (25, -25)
    latitude = (-35.282468, -35.315409)
    longitude = (149.0689, 149.156705)

    # Load data
    dc = Datacube(index=index)

    data_tiff = dc.load(
        product="ls5_nbar_scene",
        latitude=latitude,
        longitude=longitude,
        output_crs=output_crs,
        resolution=resolution,
    )
    data_zarr = dc.load(
        product="ls5_nbar_scene_zarr",
        latitude=latitude,
        longitude=longitude,
        output_crs=output_crs,
        resolution=resolution,
    )

    # compare datasets
    assert data_zarr.equals(data_tiff)
