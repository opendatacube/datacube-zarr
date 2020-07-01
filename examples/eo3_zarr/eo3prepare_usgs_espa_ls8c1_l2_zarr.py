"""
Prepare eo3 metadata for a DATASET.
A directory of files represents a DATASET.
Provide multiple DATASETS to loop through each.

Adapted from
https://github.com/GeoscienceAustralia/eo-datasets/blob/eodatasets3/eodatasets3/prepare/landsat_l1_prepare.py
"""

import logging
import os
import re
import uuid
import warnings
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union, Iterable, Dict
from bs4 import BeautifulSoup
import yaml
import click

from affine import Affine
from rasterio.crs import CRS

from eodatasets3.images import GridSpec
from eodatasets3.model import FileFormat
from eodatasets3.ui import PathPath
from .eo3_assemble import EO3DatasetAssembler

from zarr_io import ZarrIO

# label = Optional. Use as a human-readable version of the dataset ID (unique)
#         Example: f"{p.product_name}-{p.properties['landsat:landsat_scene_id']}"
# product
# - name = Taken from product_yaml
# - href = Product reference URL
# geometry = Trust these are good
# grids = Trust these are good
# properties = Mostly optional additional information populated from source metadata
#    'eo' elements should be STAC compliant
#    'landsat' elements are sensor relevant (possibly validated deep in eodatasets3 = TBA)
#    'odc' elements are helpful addition categorisation information for the ODC
#    (uppercase = MTL field. xml = alternative XML field if present)
# - datetime = DATE_ACQUIRED + SCENE_CENTER_TIME (xml: acquisition_date + scene_center_time)
# - eo:cloud_cover = CLOUD_COVER (mtl only)
# - eo:gsd = min(GRID_CELL_SIZE_*) (mtl only)
# - eo:instrument = SENSOR_ID (xml: instrument)
# - eo:platform = SPACECRAFT_ID (xml: satellite) - munged somewhere
# - eo:sun_azimuth = SUN_AZIMUTH (xml: solar_angles:azimuth)
# - eo:sun_elevation = SUN_ELEVATION (xml: 90-solar_angles:zenith)
# - landsat:collection_category = COLLECTION_CATEGORY (mtl only, other than in filenames)
# - landsat:collection_number = COLLECTION_NUMBER (mtl only)
# - landsat:geometric_rmse_model_x = GEOMETRIC_RMSE_MODEL_X (mtl only)
# - landsat:geometric_rmse_model_y = GEOMETRIC_RMSE_MODEL_Y (mtl only)
# - landsat:ground_control_points_model = GROUND_CONTROL_POINTS_MODEL (mtl only)
# - landsat:ground_control_points_version = GROUND_CONTROL_POINTS_VERSION (mtl only)
# - landsat:landsat_product_id = LANDSAT_PRODUCT_ID (xml: product_id)
# - landsat:landsat_scene_id = LANDSAT_SCENE_ID (mtl only)
# - landsat:station_id = STATION_ID (mtl only)
# - landsat:wrs_path = WRS_PATH (mtl only)
# - landsat:wrs_row = WRS_ROW (mtl only)
# - odc:dataset_version = autogen: COLLECTION_NUMBER + .0. + FILE_DATE(ymd)
# - odc:file_format = OUTPUT_FORMAT - TODO: not critical but consider check against L2 file extension
# - odc:processing_datetime = FILE_DATE (xml: level1_production_date)
# - odc:producer = "usgs.gov"
# - odc:product_family = Optional. Examples: 'level-1', 'level-2', 'ard', 'test'
# - odc:region_code = Formatted row + path
# accessories
# - relative paths to source metadata files
# lineage
# - optional. Not used for most applications outside of DEA


# Source metadata elements copied into properties:landsat:<field>
_COPYABLE_MTL_FIELDS = [
    ("metadata_file_info",
        ("landsat_scene_id", "landsat_product_id", "station_id",),
    ),
    ("product_metadata",
        ("ephemeris_type", "wrs_path", "wrs_row", "collection_category"),
    ),
    ("image_attributes",
        ("ground_control_points_version", "ground_control_points_model",
         "geometric_rmse_model_x", "geometric_rmse_model_y",
         "ground_control_points_verify", "geometric_rmse_verify",
        ),
    ),
    ("projection_parameters", ("scan_gap_interpolation",)),
]

# Static namespace to generate uuids for datacube indexing
USGS_UUID_NAMESPACE = uuid.UUID("276af61d-99f8-4aa3-b2fb-d7df68c5e28f")

# Read MTL file into a dict
MTL_PAIRS_RE = re.compile(r"(\w+)\s=\s(.*)")
def read_mtl(mtl_path: Path, root_element="l1_metadata_file") -> Dict:
    def _parse_value(s: str) -> Union[int, float, str]:
        s = s.strip('"')
        for parser in [int, float]:
            try:
                return parser(s)
            except ValueError:
                pass
        return s
    def _parse_group(
        lines: Iterable[Union[str, bytes]],
    ) -> dict:
        tree = {}
        for line in lines:
            # If line is bytes-like convert to str
            if isinstance(line, bytes):
                line = line.decode("utf-8")
            match = MTL_PAIRS_RE.findall(line)
            if match:
                key, value = match[0]
                if key == "GROUP":
                    tree[value.lower()] = _parse_group(lines)
                elif key == "END_GROUP":
                    break
                else:
                    tree[key.lower()] = _parse_value(value)
        return tree
    with mtl_path.open("r") as fp:
        tree = _parse_group(fp)
    return tree[root_element]

# Read XML file into a dict
def read_xml(xml: Path) -> dict:
    """
    Optional.
    Extract specific elements from the xml and drop them into a dict
    """
    if not xml.exists():
        warnings.warn(f'No Level-2 XML file found: {xml_path}')
        return None
    # pre-define for convenience
    d = {
        #'filenames': [],  # Not used
        'app_versions': {},  # key=xml band 'product', val=set(xml band 'app_version')
    }
    with xml.open('r') as fp:
        soup = BeautifulSoup(fp, 'xml')
        for tag in soup.find_all('band'):
            #d['filenames'].append(tag.file_name.string.strip())  # Not used
            app = tag.app_version.string.strip()
            prod = tag['product']
            if prod in d['app_versions']:
                d['app_versions'][prod].add(app)
            else:
                d['app_versions'][prod] = set((app,))
    return d


def add_measurements(assmebler: EO3DatasetAssembler, name: str, file_path: Path):
    """
    Add zarr file measurements to `EO3DatasetAssembler`.

    This replaces the call to EO3DatasetAssembler.note_measurement() which
    works for geotiffs only at this stage.
    """
    ds = ZarrIO().open_dataset(uri=file_path.as_uri())
    da = ds["band1"]
    transform = Affine(*ds.transform)
    crs = CRS.from_proj4(ds.crs)
    grid = GridSpec(da.shape, transform, crs)
    path = str(file_path.relative_to(assmebler._metadata_path.parent))
    img = da.values
    nodata = ds.nodatavals[0]
    assmebler._measurements.record_image(name, grid, path, img, nodata)


# 1. Sanity check source metadata
# 2. Populate EO3DatasetAssembler class from source metadata
# 3. Call p.done() to validate and write the dataset YAML document
def prepare_and_write(
    ds_path: Path,
    product_yaml: Path,
    output_path: Optional[Path],
    overwrite: bool,
) -> uuid.UUID:
    """
    Prepare an eo3 metadata file for a Level-2 dataset.

    Input dataset path can be a folder.
    """
    # Read (corresponding L1) MTL file
    paths = list(ds_path.rglob("*_MTL.txt"))
    if not paths:
        raise RuntimeError(f"No MTL file found for {ds_path}")
    if len(paths) > 1:
        raise RuntimeError(f"Multiple MTL files found in given dataset path {ds_path}")
    mtl_path = paths[0]
    mtl_doc = read_mtl(mtl_path)

    # Read L2 XML
    xml_path = Path(re.sub('_MTL.txt', '.xml', str(mtl_path)))
    xml_doc = read_xml(xml_path)  # issue warning and return None if not found
    if xml_doc is None:
        xml_path = None

    ## Additional product sanity-checks
    # Get and check USGS collection number
    usgs_collection_number = mtl_doc["metadata_file_info"].get("collection_number")
    if usgs_collection_number is None:
        raise NotImplementedError(
            "Dataset has no collection number: pre-collection data is not supported."
        )
    # Get and check data format
    data_format = mtl_doc["product_metadata"]["output_format"]
    if data_format.upper() != "GEOTIFF":
        raise NotImplementedError(f"Only GeoTIFF currently supported: {data_format}")
    file_format = "Zarr"
    # Get and grid cell size
    projection_params = mtl_doc["projection_parameters"]
    if (
        "grid_cell_size_thermal" in projection_params
        and "grid_cell_size_reflective" in projection_params
        and (
            projection_params["grid_cell_size_reflective"]
            != projection_params["grid_cell_size_thermal"]
        )
    ):
        raise NotImplementedError("reflective and thermal have different cell sizes")
    ground_sample_distance = projection_params["grid_cell_size_reflective"]

    ## Assemble and output
    with EO3DatasetAssembler(
        dataset_path = ds_path,
        product_yaml = product_yaml,
        metadata_path = output_path,
        overwrite = overwrite,
    ) as p:

        # Detministic ID based on USGS's product id (which changes when the scene is reprocessed by them)
        p.dataset_id = uuid.uuid5(USGS_UUID_NAMESPACE,
                                  mtl_doc["metadata_file_info"]["landsat_product_id"] + "zarr")
        p.product_uri = f"https://easi-eo.solutions/product/{p.product_name}"
        p.label = f"{p.product_name}-{mtl_doc['metadata_file_info']['landsat_scene_id']}"

        p.platform = mtl_doc["product_metadata"]["spacecraft_id"]
        p.instrument = mtl_doc["product_metadata"]["sensor_id"]
        p.product_family = "level-2"
        p.producer = "usgs.gov"
        p.datetime = "{}T{}".format(
            mtl_doc["product_metadata"]["date_acquired"],
            mtl_doc["product_metadata"]["scene_center_time"],
        )
        p.processed = mtl_doc["metadata_file_info"]["file_date"]
        p.properties["odc:file_format"] = file_format
        p.properties["eo:gsd"] = ground_sample_distance
        cloud_cover = mtl_doc["image_attributes"]["cloud_cover"]
        if cloud_cover != -1:  # Cloud cover is -1 when missing (such as TIRS-only data)
            p.properties["eo:cloud_cover"] = cloud_cover
        p.properties["eo:sun_azimuth"] = mtl_doc["image_attributes"]["sun_azimuth"]
        p.properties["eo:sun_elevation"] = mtl_doc["image_attributes"]["sun_elevation"]
        p.properties["landsat:collection_number"] = usgs_collection_number
        for section, fields in _COPYABLE_MTL_FIELDS:
            for field in fields:
                value = mtl_doc[section].get(field)
                if value is not None:
                    p.properties[f"landsat:{field}"] = value

        p.region_code = f"{p.properties['landsat:wrs_path']:03d}{p.properties['landsat:wrs_row']:03d}"
        p.dataset_version = f"{usgs_collection_number}.0.{p.processed:%Y%m%d}"

        measurement_map = p.map_measurements_to_files('T\d_(\w+).zarr')
        for measurement_name, file_location in measurement_map.items():
            logging.debug(f'Measurement map: {measurement_name} > {file_location}')
            add_measurements(p, measurement_name, file_location)

        p.add_accessory_file("metadata:landsat_mtl", mtl_path.name)

        # Ignore stac property warnings (generated in eodatasets3:properties:263)
        # eodatasets3 validates properties against a hardcoded list, which includes DEA stuff so no harm if we add our own
        warnings.filterwarnings('ignore', message='.*Unknown stac property.+landsat:l2_software_version_')

        if xml_path:
            p.add_accessory_file("metadata:landsat_xml", xml_path.name)
            for k in xml_doc['app_versions'].keys():
                s = ', '.join(xml_doc['app_versions'][k])
                p.properties['landsat:l2_software_version_' + k] = s

        return p.done()


@click.command(help=__doc__)
@click.argument(
    'datasets',
    type=PathPath(exists=True, readable=True, dir_okay=True),
    nargs=-1
)
@click.option(
    '-p', '--product', 'product_yaml',
    type=PathPath(exists=True, readable=True, dir_okay=False, file_okay=True),
    help='Product YAML to associate with the dataset',
    required=True,
)
@click.option(
    '-o', '--output-path',
    help='Write metadata files elsewhere instead of alongside each dataset',
    required=False,
    type=PathPath(exists=True, writable=True, dir_okay=True, file_okay=False),
)
@click.option(
    '--overwrite/--skip-existing',
    is_flag=True,
    default=False,
    help='Overwrite if exists (otherwise skip)',
)
def main(
    datasets: List[Path],
    product_yaml: Path,
    output_path: Optional[Path],
    overwrite: bool,
):
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO
    )
    if output_path is not None:
        output_path = output_path.resolve()  # full path
    for ds in datasets:
            output_uuid, metadata_file = prepare_and_write(
                ds_path = ds.resolve(),  # full path
                product_yaml = product_yaml,
                output_path = output_path,
                overwrite = overwrite
            )
            logging.info("Wrote dataset %s to %s", output_uuid, metadata_file)

if __name__ == "__main__":
    main()
