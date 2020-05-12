#! /usr/bin/env python3

import uuid
from datetime import datetime
from pathlib import Path

import click
import xarray as xr
import yaml

from zarr_io.zarr_io import ZarrIO


def convert_tifs_to_zarr(metadata, output_dir, group_name):
    """Convert tif dataset/agdc-metadata to zarr."""
    base_dir = metadata.parent
    with open(metadata, "r") as fh:
        meta = yaml.load(fh, Loader=yaml.SafeLoader)

    darrays = {}
    for k, v in meta["image"]["bands"].items():
        da = xr.open_rasterio(base_dir / v["path"])

        # Drop band coordinate (assumes all tifs have only 1 band)
        da = da.squeeze(drop=True)
        assert da.ndim == 2

        # Add time coordinate
        time = datetime.strptime(
            meta["extent"]["center_dt"], "%Y-%m-%d %H:%M:%S.%f"
        )
        da = da.expand_dims(dim={"time": [time]}, axis=0)

        # Add required attributes
        da.attrs["nodata"] = da.nodatavals[0]

        darrays[k] = da

    ds = xr.Dataset(darrays)

    zio = ZarrIO("file")
    zio.save_dataset(output_dir, group_name, ds)

    # Update metadata dict format and remove individual band paths
    meta["id"] = str(uuid.uuid4())
    meta["format"]["name"] = "zarr"
    for k in meta["image"]["bands"]:
        meta["image"]["bands"][k]["path"] = group_name

    # Save metadata
    zmeta = output_dir / metadata.name
    with open(zmeta, "w") as fh:
        yaml.dump(meta, fh)

    print(f"Saved metadata to: {zmeta}")


@click.command()
@click.argument("metadata", type=click.Path(exists=True))
@click.argument("output-dir", type=click.Path())
@click.argument("group-name", type=str)
def main(metadata, output_dir, group_name):
    """Convert tif dataset/agdc-metadata to zarr."""
    metadata = Path(metadata)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    convert_tifs_to_zarr(metadata, output_dir, group_name)


if __name__ == "__main__":
    main()
