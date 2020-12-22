#! /usr/bin/env python

import json

import click
import numpy as np
import xarray as xr
import yaml

from datacube_zarr.tools.zarrify import FileOrS3Path, KeyValue
from datacube_zarr.zarr_io import replace_dataset_dim

_default_zarrify_dim = "band"


@click.command()
@click.argument("zarr", type=FileOrS3Path(exists=True), required=True)
@click.argument("product", type=FileOrS3Path(exists=True), required=True)
@click.option(
    "--dim-map",
    type=KeyValue(),
    multiple=True,
    help="Dimension name mapping '<old>:<new>'.",
)
def cli(zarr, product, dim_map):
    """Replace zarr "band" dimension with product specific extra dimension."""

    pd = yaml.load(product.read_text(), Loader=yaml.SafeLoader)
    eds = pd["extra_dimensions"]
    ed_names = [ed["name"] for ed in eds]
    assert len(ed_names) == len(set(ed_names))

    replace_dims = {old: eds[ed_names.index(new)] for old, new in dim_map.items()}
    if not replace_dims:
        replace_dims[_default_zarrify_dim] = eds[0]

    for oname, ed in replace_dims.items():
        dim = xr.IndexVariable(ed["name"], np.array(ed["values"], dtype=ed["dtype"]))

        # Get number of bands / length of z dim
        band_meta_file = zarr / oname / ".zarray"
        n = json.loads(band_meta_file.read_text())["shape"][0]

        if n != len(dim):
            raise ValueError(
                f"Inconsistent dimension lengths: {oname} {n}, {ed['name']} {len(dim)}."
            )

        replace_dataset_dim(zarr.as_uri(), oname, dim)


if __name__ == "__main__":
    cli()
