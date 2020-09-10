#! /usr/bin/env python

import json
import sys

import click
import numpy as np
import xarray as xr
import yaml
from datacube_zarr.tools.zarrify import FileOrS3Path
from datacube_zarr.zarr_io import replace_dataset_dim


@click.command()
@click.argument("zarr", type=FileOrS3Path(exists=True), required=True)
@click.argument("product", type=FileOrS3Path(exists=True), required=True)
def cli(zarr, product):
    """Replace zarr "band" dimension with product specific extra dimension."""

    pd = yaml.load(product.read_text(), Loader=yaml.SafeLoader)
    ed = pd["extra_dimension"]
    dim = xr.IndexVariable(ed["name"], np.array(ed["values"], dtype=ed["dtype"]))

    # Get number of bands / length of z dim
    band_meta_file = zarr / "band/.zarray"
    n = json.loads(band_meta_file.read_text())["shape"][0]

    if n != len(dim):
        raise ValueError(
            f"Inconsistent dimension lengths: band {n}, {name} {len(dim)}."
        )

    replace_dataset_dim(zarr.as_uri(), "band", dim)


if __name__ == "__main__":
    cli()
