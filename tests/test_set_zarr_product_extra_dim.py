"""Test zarrify cli tool."""

from pathlib import Path

import pytest
import click

from datacube_zarr.tools.set_zarr_product_extra_dim import ZarrPath


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
