"""
Zarr Storage driver for ODC
Supports storage on S3 and Disk
Should be able to handle hyperspectral data when ready.
"""
import os
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import xarray as xr
from affine import Affine

import zarr
from datacube.storage import BandInfo
from datacube.utils import geometry

from .zarr_io import ZarrIO

PROTOCOL = ['file', 's3']
FORMAT = 'zarr'

RasterShape = Tuple[int, ...]
RasterWindow = Tuple[Tuple[int, int]]


def uri_split(uri: str) -> Tuple[str, str, Optional[str]]:
    """
    Splits uri into protocol, root, and group name
    """
    loc = uri.find('://')
    if loc < 0:
        return PROTOCOL[0], uri, None
    protocol = uri[:loc]
    path_str = uri[loc+3:]
    loc = path_str.rfind('/')
    group = path_str[loc+1:]
    root = path_str[:loc]
    return protocol, root, os.path.splitext(os.path.basename(group))[0]


class ZarrDataSource(object):
    class BandDataSource(object):
        def __init__(self,
                     dataset: xr.Dataset,
                     var_name: str):
            self.ds = dataset
            self._var_name = var_name
            self.da: Union[xr.DataArray, xr.Dataset] = dataset[var_name]
            self.nodata = self.da.nodata

        @property
        def crs(self) -> geometry.CRS:
            return self.da.crs

        @property
        def transform(self) -> Affine:
            return self.da.affine

        @property
        def dtype(self) -> np.dtype:
            return self.da.dtype

        @property
        def shape(self) -> RasterShape:
            return self.da.shape[1:]

        def read(self,
                 window: Optional[RasterWindow] = None,
                 out_shape: Optional[RasterShape] = None) -> Optional[np.ndarray]:
            if window is None:
                data: np.ndarray = self.da.values
            else:
                rows, cols = [slice(*w) for w in window]
                # Value of type "Union[Any, Callable[[], ValuesView[Any]]]" is not indexable
                data = self.da.values[0, rows, cols]  # type: ignore

            return data

    def __init__(self,
                 band: BandInfo,
                 protocol: Optional[str] = 's3'):
        self._band_info = band
        # Todo: Handle ODC netcdf specifics such as stacked netcdf support etc.

        # convert band.uri -> protocol, root and group
        protocol, self.root, self.group_name = uri_split(band.uri)
        self.zio = ZarrIO(protocol=protocol)

        if protocol not in PROTOCOL + ['zarr']:
            raise ValueError('Expected file:// or zarr:// url')

    @contextmanager
    def open(self) -> Generator[BandDataSource, None, None]:
        """
        Lazy open a Zarr endpoint
        """
        zarr_object = self.zio.open_dataset(root=self.root,
                                            group_name=self.group_name, relative=False)
        yield ZarrDataSource.BandDataSource(dataset=zarr_object,
                                            var_name=self._band_info.name)


class ZarrReaderDriver(object):
    def __init__(self,
                 protocol: Optional[str] = 's3'):
        self.name = 'ZarrReader'
        self.protocol = protocol
        self.protocols = PROTOCOL + ['zarr']
        self.formats = [FORMAT]

    def supports(self,
                 protocol: str,
                 fmt: str) -> bool:
        return (protocol in self.protocols and
                fmt in self.formats)

    def new_datasource(self,
                       band: BandInfo) -> ZarrDataSource:
        return ZarrDataSource(band, self.protocol)


def s3_reader_driver_init() -> ZarrReaderDriver:
    return ZarrReaderDriver()


def file_reader_driver_init() -> ZarrReaderDriver:
    return ZarrReaderDriver(protocol='file')


class ZarrWriterDriver(object):
    def __init__(self,
                 protocol: Optional[str] = 's3'):
        self.zio = ZarrIO(protocol=protocol)

    @property
    def aliases(self) -> List:
        if self.zio.protocol == 's3':
            return ['zarr s3']
        elif self.zio.protocol == 'file':
            return ['zarr file']
        else:
            return []

    @property
    def format(self) -> str:
        return FORMAT

    @property
    def uri_scheme(self) -> str:
        return self.zio.protocol

    def write_dataset_to_storage(self,
                                 dataset: xr.Dataset,
                                 filename: str,
                                 global_attributes: Optional[dict] = None,
                                 variable_params: Optional[dict] = None,
                                 storage_config: Optional[dict] = None,
                                 **kwargs: str) -> Dict:
        if storage_config:
            root = storage_config['root']
        else:
            raise ValueError('storage/root not defined in ingest yaml')
        filename = os.path.splitext(os.path.basename(filename))[0]

        # Flattening atributes: Zarr doesn't allow dicts
        for var_name in dataset.data_vars:
            data_var = dataset.data_vars[var_name]
            if 'spectral_definition' in data_var.attrs:
                spectral_definition = data_var.attrs.pop('spectral_definition', None)
                data_var.attrs['dc_spectral_definition_response'] = spectral_definition['response']
                data_var.attrs['dc_spectral_definition_wavelength'] = spectral_definition['wavelength']

        # Renaming units: units is a reserved name in Xarray coordinates
        for var_name in dataset.coords:
            coord_var = dataset.coords[var_name]
            if 'units' in coord_var.attrs:
                units = coord_var.attrs.pop('units', None)
                coord_var.attrs['dc_units'] = units

        # Should be a directory but actually get passed a file, which becomes a directory.
        metadata = self.zio.save_dataset_to_zarr(root=root,
                                                 dataset=dataset,
                                                 filename=filename,
                                                 global_attributes=global_attributes,
                                                 variable_params=variable_params,
                                                 storage_config=storage_config)

        # extra metadata to be stored in database
        return metadata


def s3_writer_driver_init() -> ZarrWriterDriver:
    return ZarrWriterDriver()


def file_writer_driver_init() -> ZarrWriterDriver:
    return ZarrWriterDriver(protocol='file')
