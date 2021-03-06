#! /usr/bin/env python

"""
Create an `agdc-metadata.xml` file for a Landsat zarr dataset on file or S3

Modified from GA's `galsprepare.py`.
"""

import re
import uuid
from datetime import timedelta
from pathlib import Path
from typing import Generator, Optional
from xml.etree import ElementTree

import click
import osgeo
import xarray as xr
import yaml
import zarr
from dateutil import parser
from osgeo import osr
from s3path import S3Path

from datacube_zarr import ZarrIO

_STATIONS = {
    '023': 'TKSC',
    '022': 'SGS',
    '010': 'GNC',
    '011': 'HOA',
    '012': 'HEOC',
    '013': 'IKR',
    '014': 'KIS',
    '015': 'LGS',
    '016': 'MGR',
    '017': 'MOR',
    '032': 'LGN',
    '019': 'MTI',
    '030': 'KHC',
    '031': 'MLK',
    '018': 'MPS',
    '003': 'BJC',
    '002': 'ASN',
    '001': 'AGS',
    '007': 'DKI',
    '006': 'CUB',
    '005': 'CHM',
    '004': 'BKT',
    '009': 'GLC',
    '008': 'EDC',
    '029': 'JSA',
    '028': 'COA',
    '021': 'PFS',
    '020': 'PAC',
}

_PRODUCTS = {'NBART': 'nbart', 'NBAR': 'nbar', 'PQ': 'pqa', 'FC': 'fc'}


def band_name(name):
    position = name.rfind('_')
    if position == -1:
        raise ValueError(f'Unexpected zarr dataset found: {name}')
    if re.match(r"[Bb]\d+", name[position + 1 :]):
        band_number = name[position + 2 : position + 3]
    elif name[position + 1 :].startswith('1111111111111100'):
        band_number = 'pqa'
    else:
        band_number = name[position + 1 :]
    return band_number


def get_projection(ds):
    left, bottom, right, top = ds.extent.boundingbox
    proj = {
        "spatial_reference": ds.geobox.crs.wkt,
        "geo_ref_points": {
            'ul': {'x': left, 'y': top},
            'ur': {'x': right, 'y': top},
            'll': {'x': left, 'y': bottom},
            'lr': {'x': right, 'y': bottom},
        },
    }
    return proj


def get_coords(geo_ref_points, spatial_ref):
    spatial_ref = osr.SpatialReference(spatial_ref)
    if int(osgeo.__version__[0]) >= 3:
        spatial_ref.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    t = osr.CoordinateTransformation(spatial_ref, spatial_ref.CloneGeogCS())

    def transform(p):
        lon, lat, z = t.TransformPoint(p['x'], p['y'])
        return {'lon': lon, 'lat': lat}

    return {key: transform(p) for key, p in geo_ref_points.items()}


def populate_coord(doc):
    proj = doc['grid_spatial']['projection']
    doc['extent']['coord'] = get_coords(proj['geo_ref_points'], proj['spatial_reference'])


def crazy_parse(timestr):
    try:
        return parser.parse(timestr)
    except ValueError:
        if not timestr[-2:] == "60":
            raise
        return parser.parse(timestr[:-2] + '00') + timedelta(minutes=1)


def get_groups_with_arrays(group: zarr.Group) -> Generator[str, None, None]:
    """Find any arrays within zarr heirarchy."""
    if list(group.arrays()):
        yield group.name
    for _, g in group.groups():
        yield from get_groups_with_arrays(g)


def get_zarr_objs(path: Path):
    """Find subdir which are zarr obj roots."""
    zio = ZarrIO()
    for r in path.iterdir():
        if next(r.glob(".zgroup"), None) is not None:
            root_group = zarr.group(zio.get_root(r.as_uri()))
            for g in get_groups_with_arrays(root_group):
                yield r, g if g != "/" else None


def load_zarr_dataset(path: Path, group: str) -> xr.DataArray:
    """Open zarr datasets on s3 or file."""
    zio = ZarrIO()
    da = zio.open_dataset(uri=path.as_uri())
    return da


def prep_dataset(fields, path):
    meta = path / "metadata.xml"
    doc = ElementTree.fromstring(meta.read_text())
    aos = crazy_parse(doc.findall("./ACQUISITIONINFORMATION/EVENT/AOS")[0].text)
    los = crazy_parse(doc.findall("./ACQUISITIONINFORMATION/EVENT/LOS")[0].text)
    start_time = crazy_parse(doc.findall("./EXEXTENT/TEMPORALEXTENTFROM")[0].text)
    end_time = crazy_parse(doc.findall("./EXEXTENT/TEMPORALEXTENTTO")[0].text)

    scene = path / "scene01"
    zarrs = list(get_zarr_objs(scene))
    zarr_paths = {
        band_name(r.stem): {
            "path": "#".join([str(r.relative_to(path))] + ([g] if g else [])),
            "layer": "array",
        }
        for r, g in zarrs
    }

    doc = {
        'id': str(uuid.uuid4()),
        'processing_level': fields["level"],
        'product_type': _PRODUCTS[fields["type"]],
        'creation_dt': str(aos),
        'platform': {'code': "LANDSAT_" + fields["vehicle"][2]},
        'instrument': {'name': fields["instrument"]},
        'acquisition': {
            'groundstation': {'code': _STATIONS[fields["groundstation"]]},
            'aos': str(aos),
            'los': str(los),
        },
        'extent': {
            'from_dt': str(start_time),
            'to_dt': str(end_time),
            'center_dt': str(start_time + (end_time - start_time) // 2),
        },
        'format': {'name': 'zarr'},
        'grid_spatial': {'projection': get_projection(load_zarr_dataset(*zarrs[0]))},
        'image': {
            'satellite_ref_point_start': {
                'x': int(fields["path"]),
                'y': int(fields["row"]),
            },
            'satellite_ref_point_end': {
                'x': int(fields["path"]),
                'y': int(fields["row"]),
            },
            'bands': zarr_paths,
        },
        'lineage': {'source_datasets': {}},
    }
    populate_coord(doc)
    return doc


def dataset_folder(fields):
    fmt_str = (
        "{vehicle}_{instrument}_{type}_{level}_GA{type}{product}-"
        "{groundstation}_{path}_{row}_{date}"
    )
    return fmt_str.format(**fields)


def prepare_datasets(nbar_path, pq_path=None, fc_path=None):
    fields = re.match(
        (
            r"(?P<vehicle>LS[578])"
            r"_(?P<instrument>OLI_TIRS|OLI|TIRS|TM|ETM)"
            r"_(?P<type>NBAR)"
            r"_(?P<level>P54)"
            r"_GA(?P=type)(?P<product>[0-9][0-9])"
            r"-(?P<groundstation>[0-9]{3})"
            r"_(?P<path>[0-9]{3})"
            r"_(?P<row>[0-9]{3})"
            r"_(?P<date>[12][0-9]{7})"
            "$"
        ),
        nbar_path.stem,
    ).groupdict()

    nbar = prep_dataset(fields, nbar_path)

    fields.update({'type': 'PQ', 'level': 'P55'})
    pq_path = (pq_path or nbar_path.parent).joinpath(dataset_folder(fields))
    if not pq_path.exists():
        return [(nbar, nbar_path)]

    pq = prep_dataset(fields, pq_path or nbar_path.parent)
    pq['lineage']['source_datasets'] = {nbar['id']: nbar}

    fields.update({'type': 'FC', 'level': 'P54'})
    fc_path = (fc_path or nbar_path.parent).joinpath(dataset_folder(fields))
    if not fc_path.exists():
        return (nbar, nbar_path), (pq, pq_path)

    fc = prep_dataset(fields, fc_path or nbar_path.parent)
    fc['lineage']['source_datasets'] = {nbar['id']: nbar, pq['id']: pq}

    return (nbar, nbar_path), (pq, pq_path), (fc, fc_path)


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
            if value.startswith("file://"):
                value = value[7:]
            path = Path(value).resolve()

        if self.exists and not path.exists:
            raise ValueError(f"{path.as_uri()} does not exist.")

        return path


@click.command()
@click.argument("dataset", type=FileOrS3Path(exists=True), required=True)
def main(dataset: Path):
    """Create an `agdc-metadata.xml` file for a LS5 zarr dataset."""
    documents = prepare_datasets(dataset)
    if documents:
        for dataset, path in documents:
            yaml_path = path / 'agdc-metadata.yaml'
            yaml_path.write_text(yaml.dump(dataset))
            print(yaml_path.as_uri())


if __name__ == "__main__":
    main()
