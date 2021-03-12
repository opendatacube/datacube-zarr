#! /usr/bin/env python

from pathlib import Path
from typing import List, Optional, Tuple

import click
import numpy as np
import xarray as xr
import yaml

from datacube_zarr.tools.zarrify import FileOrS3Path, KeyValue
from datacube_zarr.zarr_io import ZarrIO, replace_dataset_dim

_default_zarrify_dim = "band"


class ZarrPath(click.ParamType):
    """A click param for any file or s3 zarr path.

    Returns a pathlib Path + group tuple.
    """

    name = "ZarrPath"

    def __init__(self, exists: bool = False):
        self.exists = exists

    def convert(
        self, value: str, param: Optional[click.Parameter], ctx: Optional[click.Context]
    ) -> Tuple[Path, str]:
        """Convert file/s3 path str to pathlib (Path, group)."""
        if "#" in value:
            pstr, group = value.split("#")
        else:
            pstr, group = value, ""

        path = FileOrS3Path(exists=self.exists).convert(pstr, param, ctx)

        if self.exists and not (path / group).exists():
            self.fail(f"{path.as_uri()}#{group} does not exist.")

        return (path, group)


@click.command()
@click.argument("zarr", type=ZarrPath(exists=True), required=True)
@click.argument("product_def", type=FileOrS3Path(exists=True), required=True)
@click.option("--name", type=str, help="Product name")
@click.option(
    "--dim-map",
    type=KeyValue(),
    multiple=True,
    help="Dimension name mapping '<old>:<new>'.",
)
def cli(
    zarr: Tuple[Path, str], product_def: Path, name: str, dim_map: List[Tuple]
) -> None:
    """Replace zarr "band" dimension with product specific extra dimension."""
    zpath, group = zarr
    zarr_uri = zpath.as_uri() + (f"#{group}" if group else "")

    # Load extra dimensions from product definition
    pds = yaml.load_all(product_def.read_text(), Loader=yaml.SafeLoader)
    if name is not None:
        pds = (pd for pd in pds if pd["name"] == name)

    try:
        pd = next(pds)
    except StopIteration:
        raise click.ClickException("No matching product definition found.")

    if "extra_dimensions" not in pd:
        raise click.ClickException(
            f"Product definition for '{pd['name']}' has no 'extra_dimensions'."
        )

    eds = pd["extra_dimensions"]
    ed_names = [ed["name"] for ed in eds]
    assert len(ed_names) == len(set(ed_names))

    # Mapping from zarr dim names to extra dimensions
    replace_dims = {old: eds[ed_names.index(new)] for old, new in dim_map}
    if not replace_dims:
        replace_dims[_default_zarrify_dim] = eds[0]

    # Replace zarr dims with new extra dimension data
    for oname, ed in replace_dims.items():
        dim = xr.IndexVariable(ed["name"], np.array(ed["values"], dtype=ed["dtype"]))

        # Get number of bands / length of z dim
        n = len(ZarrIO().open_dataset(zarr_uri)[oname])

        if n != len(dim):
            raise ValueError(
                f"Inconsistent dimension lengths: {oname} {n}, {ed['name']} {len(dim)}."
            )

        replace_dataset_dim(zarr_uri, oname, dim)


if __name__ == "__main__":
    cli()
