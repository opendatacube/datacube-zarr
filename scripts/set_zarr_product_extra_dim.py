#! /usr/bin/env python

import json

import click
import numpy as np
import xarray as xr
import yaml

from datacube_zarr.tools.zarrify import FileOrS3Path
from datacube_zarr.zarr_io import replace_dataset_dim


@click.command()
@click.argument("zarr", type=FileOrS3Path(exists=True), required=True)
@click.argument("product_def", type=FileOrS3Path(exists=True), required=True)
@click.option("--name", type=str, help="Product name")
def cli(zarr, product_def, name):
    """Replace zarr "band" dimension with product specific extra dimension."""

    pds = yaml.load_all(product_def.read_text(), Loader=yaml.SafeLoader)
    if name is not None:
        pds = (pd for pd in pds if pd["name"] == name)

    try:
        pd = next(pds)
    except StopIteration:
        raise click.ClickException("No matching product definition found.")

    if "extra_dimension" not in pd:
        raise click.ClickException(
            f"Product definition for '{pd['name']}' has no 'extra_dimension'."
        )

    ed = pd["extra_dimension"]
    dim = xr.IndexVariable(ed["name"], np.array(ed["values"], dtype=ed["dtype"]))

    # Get number of bands / length of z dim
    band_meta_file = zarr / "band/.zarray"
    n = json.loads(band_meta_file.read_text())["shape"][0]

    if n != len(dim):
        raise ValueError(
            f"Inconsistent dimension lengths: band {n}, {ed['name']} {len(dim)}."
        )

    replace_dataset_dim(zarr.as_uri(), "band", dim)


if __name__ == "__main__":
    cli()
