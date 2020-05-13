#! /usr/bin/env python

import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple
from xml.etree import ElementTree

import click
import xarray as xr
from dateutil.parser import parse as date_parse
from s3path import S3Path

from zarr_io import ZarrIO


class KeyValue(click.ParamType):
    """A click param for any key/value pairs."""

    name = "KeyValue"

    def __init__(self, key_type: type = str, val_type: type = str, sep: str = ":"):
        self.key_type = key_type
        self.val_type = val_type
        self.sep = sep

    def convert(
        self, value: str, param: Optional[click.Parameter], ctx: Optional[click.Context]
    ) -> Tuple:
        """Convert key:valueto tuple."""
        k, v = value.split(self.sep, 1)
        return self.key_type(k), self.val_type(v)


class FileOrS3Path(click.ParamType):
    """A click param for any file or s3 path."""

    name = "FileOrS3Path"

    def __init__(self, exists: bool = False):
        self.exists = exists

    def convert(
        self, value: str, param: Optional[click.Parameter], ctx: Optional[click.Context]
    ) -> Path:
        """Convert file/s3 path str to pathlib Path."""
        if value.startswith("s3:/"):
            path = S3Path(value[4:])
        else:
            path = Path(value)

        if self.exists and not path.exists:
            raise ValueError(f"{path.as_uri()} does not exist.")

        return path


def save_dataset_to_zarr(
    ds: xr.Dataset, root: Path, group: str, **kwargs: Any
) -> None:
    """Save an xarray dataset to s3 or file in Zarr format."""
    if root.as_uri().startswith("s3://"):
        protocol = "s3"
        root_ = root.as_uri()
    else:
        protocol = "file"
        root_ = str(root)

    zio = ZarrIO(protocol)
    zio.save_dataset(root=root_, group_name=group, dataset=ds, **kwargs)


def convert_ls5_scene(
    scene_path: Path, zarr_dir: Path, chunks: Optional[Dict[str, int]]
) -> None:
    """Convert scene into Zarr format."""
    group = scene_path.parts[-1]
    meta, scene_attr = get_ls5_metadata(scene_path)
    scene_time = scene_attr.pop("time")

    darrays: Dict[str, xr.DataArray] = {}
    tifs = sorted(scene_path.glob("**/*.tif"))
    for t in tifs:
        da = xr.open_rasterio(t.as_uri())

        # Drop band coordinate (assumes all tifs have only 1 band)
        da = da.squeeze(drop=True)
        assert da.ndim == 2

        # Add time coordinate
        da = da.expand_dims(dim={"time": [scene_time]}, axis=0)

        # Add nodata value
        da.attrs["nodata"] = da.nodatavals[0]
        da.attrs.update(scene_attr)

        # Get a band name from file
        band = get_ls5_band_name(t.stem)
        assert band not in darrays
        darrays[band] = da

    # Save dataset
    ds = xr.Dataset(darrays)
    save_dataset_to_zarr(ds, zarr_dir, group, chunks=chunks)

    # Copy metadata
    meta_cp = zarr_dir / meta.name
    assert meta_cp.parent.exists()
    meta_cp.write_text(meta.read_text())


def get_ls5_band_name(name: str):
    """Extract landsat band name from geotif filename."""
    i = name.rfind('_')
    if i == -1:
        raise ValueError(f"Unexpected tif image in eods: {name}")
    if re.match(r"[Bb]\d+", name[i + 1:]):
        band = name[i + 2:i + 3]
    elif name[i + 1:].startswith('1111111111111100'):
        band = 'pqa'
    else:
        band = name[i + 1:]
    return band


def get_ls5_metadata(scene_path: Path) -> Tuple[Path, Dict]:
    """Copy metadata file and extract scene time."""

    def _find_datetime(tree: ElementTree.Element, path: str) -> datetime:
        """Find a datetime in an XML tree."""
        elem = tree.find("./EXEXTENT/TEMPORALEXTENTFROM")
        if elem is not None:
            dt = date_parse(str(elem.text))
        else:
            raise ValueError("Could not find datetime within 'metadata.xml'.")
        return dt

    try:
        meta = next(scene_path.glob("metadata.xml"))
    except StopIteration:
        raise FileNotFoundError(f"Can't find metadata.xml within {scene_path}.")

    tree = ElementTree.fromstring(meta.read_text())
    start_time = _find_datetime(tree, "./EXEXTENT/TEMPORALEXTENTFROM")
    end_time = _find_datetime(tree, "./EXEXTENT/TEMPORALEXTENTTO")
    attr = {"time": start_time + (end_time - start_time) // 2}
    return meta, attr


def get_ls5_dataset_dirs(p: Path) -> Generator[Path, None, None]:
    """Find directories with 'metadata.xml' file."""
    subp = list(p.iterdir())
    if p / "metadata.xml" in subp:
        yield p
    else:
        yield from (p for s in subp if s.is_dir() for p in get_ls5_dataset_dirs(s))


@click.command()
@click.argument("dataset", type=FileOrS3Path(exists=True), required=True)
@click.option("--outdir", type=FileOrS3Path(), required=True)
@click.option("--chunks", type=KeyValue(val_type=int), multiple=True)
def main(dataset: Path, outdir: Path, chunks: Optional[List[Tuple[str, int]]]) -> None:
    """Convert (LS5) datasets to Zarr format."""
    chunks_ = dict(chunks) if chunks else None
    for d in get_ls5_dataset_dirs(dataset):
        outdir_ = outdir / d.stem
        convert_ls5_scene(d, outdir_, chunks=chunks_)
        print(outdir_)


if __name__ == "__main__":
    main()
