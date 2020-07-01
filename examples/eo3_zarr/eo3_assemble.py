import os
import re
import uuid
import warnings
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
import yaml

from eodatasets3 import serialise, validate, images
from eodatasets3.validate import Level
from eodatasets3.properties import EoFields
from eodatasets3.images import GridSpec, MeasurementRecord
from eodatasets3.model import StacPropertyView, DatasetDoc, ProductDoc, AccessoryDoc
import rasterio
from rasterio.crs import CRS

# Adapted from
# https://github.com/GeoscienceAustralia/eo-datasets/blob/eodatasets3/eodatasets3/assemble.py

METADATA_NAME = 'odc-metadata.yaml'

class EO3DatasetAssembler(EoFields):
    def __init__(
        self,
        dataset_path: Path,
        product_yaml: Path,
        metadata_path: Optional[Path] = None,
        overwrite: bool = False,
    ) -> None:
        """
        Assemble eo3 metadata for a dataset.

        :param dataset_path:
            Path to the dataset. Dataset can be a single file or a directory of files.
            A directory of files represents one dataset.
            Files should readable by GDAL to allow generation of the grid specifications.

        :param product_yaml:
            Path to the corresponding product YAML. The product name is used and the
            measurements must correspond to file(s).

        :param metadata_path:
            Optional. Specify the output dataset YAML file. Default is to create it next
            to the dataset file(s).

        :param overwrite:
            Optional. Whether to overwrite an existing dataset YAML file. Default is False.
        """

        self._dataset_location = dataset_path.resolve()  # absolute path
        self._product_yaml = product_yaml
        self._metadata_path = metadata_path
        self._overwrite = overwrite

        self._dataset_id = uuid.uuid4()  # default
        self._label = None
        self._product_name = self.get_product_name()
        self._product_uri = None
        self._props = StacPropertyView()
        self._accessories: Dict[str, Path] = {}
        self._measurements = MeasurementRecord()
        self._user_metadata = dict()
        self._software_versions: List[Dict] = []
        self._lineage: Dict[str, List[uuid.UUID]] = defaultdict(list)

        self._set_metadata_path()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Any necessary clean up
        pass

    def _set_metadata_path(self):
        """
        Metadata_path will be resolved to its absolute path.
        If metadata_path is a directory then the default metadata name will be used.
        If metadata_path has not been defined then a default is defined adjacent
        to files in dataset_location.
        """
        meta = None
        if self._metadata_path is not None:
            meta = self._metadata_path.resolve()  # absolute path
            if meta.is_dir():
                meta = meta / METADATA_NAME
        else:
            if self._dataset_location.is_dir():
                meta = self._dataset_location / METADATA_NAME
            else:
                meta = self._dataset_location.parent / METADATA_NAME
        self._metadata_path = meta

    # Should it be _dataset_id? > check doc writer
    @property
    def dataset_id(self):
        return self._dataset_id

    @dataset_id.setter
    def dataset_id(self, val: str):
        self._dataset_id = val

    @property
    def label(self) -> str:
        return self._label

    @label.setter
    def label(self, val: str):
        self._label = val

    @property
    def product_name(self) -> str:
        return self._product_name

    @property
    def product_uri(self) -> str:
        return self._product_uri

    @product_uri.setter
    def product_uri(self, val: str):
        self._product_uri = val

    @property
    def properties(self) -> StacPropertyView:
        return self._props

    def add_accessory_file(self, name: str, path: Path):
        """
        Record a reference to an additional file.

        :param name: identifying name, eg 'metadata:mtl'
        :param path: local path to file.
        """
        existing_path = self._accessories.get(name)
        if existing_path is not None and existing_path != path:
            raise ValueError(f"Duplicate accessory name {name!r}")
        self._accessories[name] = path

    def note_measurement(
        self,
        measurement_name: str,
        file_path: Path,
        expand_valid_data=True,
        relative_to_metadata=True,
    ):
        """
        Reference a measurement from its existing file path.
        (no data is copied, but Geo information is read from it.)

        :param measurement_name:
        Measurement name corresponding to a product measurement
        :param file_path:
        Path to data file for this measurement
        :param expand_valid_data:
        Not sure, hangover from eodatasets3
        :param relative_to_metadata:
        File paths in the dataset doc will be written relative to metadata_path
        """
        # relative path to file
        read_path = file_path
        if relative_to_metadata:
            read_path = os.path.relpath(file_path, self._metadata_path.parent)

        with rasterio.open(file_path) as ds:
            # TODO: fix for multi-band files
            ds: DatasetReader
            if ds.count != 1:
                raise NotImplementedError("TODO: Only single-band files currently supported")
            self._measurements.record_image(
                measurement_name,
                images.GridSpec.from_rio(ds),
                read_path,
                ds.read(1),
                nodata=ds.nodata,
                expand_valid_data=expand_valid_data,
            )

    def get_product_name(self) -> str:
        """
        Return the product name from the product yaml
        """
        with self._product_yaml.open() as f:
            y = yaml.load(f, Loader=yaml.FullLoader)
            return y['name']

    def get_product_measurements(self) -> list:
        """
        Return list of (measurement, alias, ..) tuples
        """
        measurements = []  # list of tuples (measurement name, alias, ...)
        with self._product_yaml.open() as f:
            y = yaml.load(f, Loader=yaml.FullLoader)
            for m in y['measurements']:
                t = [m['name']]
                if 'aliases' in m:
                    t.extend(m['aliases'])
                measurements.append(tuple(t))
        return measurements

    def map_measurements_to_files(
        self,
        band_regex: str,
        supplementary: dict = None,
    ) -> dict:
        """
        Return dict of {measurement names: filenames} for matching files in dataset_location.

        Each measurement in the product yaml must have a corresponding file.
        Loop through the list of product measurements
        - Compare filename band_id (from BAND_REGEX) with tuples of measurement names and aliases
        - If found key is first element of tuple (measurement name) and value is filename
        - If not found try band_id key in SUPPLEMENTARY

        :param band_regex:
        Regular expression string to apply to each file in dataset_location.
        Match.group(1) should be the band ID used to select the corresponding measurement

        :param supplementary:
        Dict mapping any band IDs (from band_regex) to measurement names.
        Use where the unique band ID does not directly match a measurement name.
        """
        # Matching is done by interesection of sets, where a single common element indicates a successful match
        measurement2file = {}
        supplementary_set = set()
        if supplementary:
            supplementary_set = set(supplementary.keys())

        # Match band_ids to filenames
        p = re.compile(band_regex)
        band_ids = {}
        if self._dataset_location.is_dir():
            for filename in self._dataset_location.iterdir():
                m = p.search(str(filename.name))
                if not m:
                    continue
                band_ids[m.group(1)] = filename
        else:
            filename = self._dataset_location.name
            m = p.search(str(filename))
            if m:
                band_ids[m.group(1)] = filename
        band_set = set(band_ids.keys())
        if len(band_set) == 0:
            raise RuntimeError('No matching files found for regex: {band_regex}')

        # Match measurement names to band_ids (> filenames)
        for mtuple in self.get_product_measurements():
            common = set(mtuple) & band_set
            if len(common) == 1:
                measurement2file[mtuple[0]] = band_ids[common.pop()]
                continue
            # Try supplementary
            common = supplementary_set & band_set
            if len(common) == 1:
                c = common.pop()
                measurement2file[supplementary[c]] = band_ids[c]
                continue
            raise RuntimeError(f'No unique match for measurements {mtuple} in files: {self._dataset_location}')
        return measurement2file

    def done(
        self,
        validate_correctness: bool = True,
        sort_measurements: bool = True
    ) -> Tuple[uuid.UUID, Path]:
        """
        Validate and write the dataset metadata doc.

        :param validate_correctness:
        Run the eo3-validator on the resulting metadata
        :param sort_measurements:
        Order measurements alphabetically (instead of insert-order)

        :returns: The id and final path to the dataset metadata file.
        """
        crs, grid_docs, measurement_docs = self._measurements.as_geo_docs()

        # Add layer to specify zarr variable to load
        # This should be added to `MeasurementRecord.record_image()`
        for md in measurement_docs.values():
            md.layer = "band1"

        if measurement_docs and sort_measurements:
            measurement_docs = dict(sorted(measurement_docs.items()))

        valid_data = self._measurements.consume_and_get_valid_data()
        # Avoid the messiness of different empty collection types.
        # (to have a non-null geometry we'd also need non-null grids and crses)
        if valid_data.is_empty:
            valid_data = None

        dataset = DatasetDoc(
            id=self._dataset_id,
            label=self.label,
            product=ProductDoc(
                name=self._product_name, href=self._product_uri
            ),
            crs=self._crs_str(crs) if crs is not None else None,
            geometry=valid_data,
            grids=grid_docs,
            properties=self.properties,
            accessories={
                name: AccessoryDoc(path, name=name)
                for name, path in self._accessories.items()
            },
            measurements=measurement_docs,
            lineage=self._lineage,
        )

        doc = serialise.to_formatted_doc(dataset)
        self._write_yaml(doc, self._metadata_path)

        if validate_correctness:
            for m in validate.validate_dataset(doc):
                if m.level in (Level.info, Level.warning):
                    warnings.warn(str(m))
                elif m.level == Level.error:
                    raise m
                else:
                    raise RuntimeError(
                        f"Internal error: Unhandled type of message level: {m.level}"
                    )

        assert self._metadata_path.exists()
        return dataset.id, self._metadata_path

    def _crs_str(self, crs: CRS) -> str:
        return f"epsg:{crs.to_epsg()}" if crs.is_epsg_code else crs.to_wkt()

    def _write_yaml(self, doc, path, allow_external_paths=False):
        # documents.make_paths_relative(
        #     doc, path.parent, allow_paths_outside_base=allow_external_paths
        # )
        serialise.dump_yaml(path, doc)
        # self._checksum.add_file(path)