"""Test zarrify cli tool."""

from pathlib import Path

import pytest
import click
from click.testing import CliRunner

from datacube_zarr.tools.set_zarr_product_extra_dim import ZarrPath, cli


def test_zarr_path(zarr_with_group):
    """Test ZarrPath click param."""
    p, g = ZarrPath(exists=True).convert(zarr_with_group, None, None)
    p_, g_ = zarr_with_group.split("#")
    assert p == Path(p_[len("file:") :])
    assert g == g_


def test_zarr_path_bad_group(zarr_with_group):
    """Test ZarrPath click param."""
    zarr_bad_group = zarr_with_group.split("#")[0] + "#bad_group_name"
    with pytest.raises(click.BadParameter):
        ZarrPath(exists=True).convert(zarr_bad_group, None, None)


TEST_PROD_DEF_3D = """
name: test_prod
description: Test product definition file
metadata_type: eo3
metadata:
  product:
    name: test_prod
extra_dimensions:
- name: z
  values: [5, 10, 15, 20, 25]
  dtype: float64
measurements:
- name: meas_z
  extra_dim:
    dimension: z
    measurement_map: ['5', '10', '15', '20', '25']
    alias_map:
    - [band_1]
    - [band_2]
    - [band_3]
    - [band_4]
    - [band_5]
  dtype: float32
  nodata: -9999.0
"""


@pytest.fixture()
def mock_product_def_3d(tmp_path):
    pd_file = tmp_path / "test_prod_3d.yaml"
    pd_file.write_text(TEST_PROD_DEF_3D)
    return pd_file


TEST_PROD_DEF_2D = """
name: test_prod
description: Test product definition file
metadata_type: eo3
metadata:
  product:
    name: test_prod
measurements:
- name: meas_z
  dtype: float32
  nodata: -9999.0
"""


@pytest.fixture()
def mock_product_def_2d(tmp_path):
    pd_file = tmp_path / "test_prod_3d.yaml"
    pd_file.write_text(TEST_PROD_DEF_2D)
    return pd_file


@pytest.fixture(scope="session")
def dummy_zarr_path(tmp_path_factory):
    zarr_dir = tmp_path_factory.mktemp("gedi_3d_zarrs")
    dummy_zarr = zarr_dir / "dummt.zarr"
    dummy_zarr.touch()
    return dummy_zarr


def test_zarr_extra_dim(tmp_3d_zarr, mock_product_def_3d):
    runner = CliRunner()
    args = ["--name", "test_prod", str(tmp_3d_zarr), str(mock_product_def_3d)]
    res = runner.invoke(cli, args)
    assert res.exit_code == 0, res.stdout


def test_zarr_extra_dim_bad_name(mock_product_def_3d, dummy_zarr_path):
    runner = CliRunner()
    args = ["--name", "bad_prod", str(dummy_zarr_path), str(mock_product_def_3d)]
    res = runner.invoke(cli, args)
    assert res.exit_code == 1
    assert "No matching product definition found." in res.stdout


def test_zarr_extra_dim_bad_pd(mock_product_def_2d, dummy_zarr_path):
    runner = CliRunner()
    args = ["--name", "test_prod", str(dummy_zarr_path), str(mock_product_def_2d)]
    res = runner.invoke(cli, args)
    assert res.exit_code == 1
    assert "Product definition for 'test_prod' has no 'extra_dimensions'." in res.stdout


TEST_PROD_DEF_3D_LEN2 = """
name: test_prod
description: Test product definition file
metadata_type: eo3
metadata:
  product:
    name: test_prod
extra_dimensions:
- name: z
  values: [5, 10]
  dtype: float64
measurements:
- name: meas_z
  extra_dim:
    dimension: z
    measurement_map: ['5', '10']
  dtype: float32
  nodata: -9999.0
"""


@pytest.fixture()
def mock_product_def_3d_len2(tmp_path):
    pd_file = tmp_path / "test_prod_3d_len2.yaml"
    pd_file.write_text(TEST_PROD_DEF_3D_LEN2)
    return pd_file


def test_zarr_extra_dim_bad_dim_len(tmp_3d_zarr, mock_product_def_3d_len2):
    runner = CliRunner()
    args = ["--name", "test_prod", str(tmp_3d_zarr), str(mock_product_def_3d_len2)]
    res = runner.invoke(cli, args)
    assert res.exit_code == 1
    assert "Inconsistent dimension lengths: band 5, z 2." in res.stdout
