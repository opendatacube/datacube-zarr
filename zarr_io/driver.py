"""
Zarr Storage driver for ODC
Supports storage on S3 and Disk
Should be able to handle hyperspectral data when ready.
"""
import itertools
from contextlib import contextmanager
from typing import Dict, Generator, List, Optional, Tuple

import numpy as np
import xarray as xr
from affine import Affine
from datacube.storage import BandInfo
from datacube.utils import geometry
from datacube.utils.math import num2numpy

from .utils.uris import uri_join, uri_split
from .zarr_io import ZarrIO

PROTOCOL = ['file', 's3']
FORMAT = 'zarr'

RasterShape = Tuple[int, ...]
RasterWindow = Tuple[Tuple[int, int]]


class ZarrDataSource(object):
    class BandDataSource(object):
        def __init__(
            self,
            dataset: xr.Dataset,
            var_name: str,
            time_idx: Optional[int],
            no_data: Optional[float],
        ):
            """
            Initialises the BandDataSource class.

            The BandDataSource class to read array slices out of the xr.Dataset.

            :param xr.Dataset dataset: The xr.Dataset
            :param str var_name: The variable name of the xr.DataArray
            :param int time_idx: The time index override if known
            :param float no_data: The no data value if known
            """
            self.ds = dataset
            self._var_name = var_name
            self.da = dataset.data_vars[var_name]
            self._nodata = (
                self.da.nodata
                if 'nodata' in self.da.attrs and self.da.nodata
                else no_data
            )
            if not self._nodata:
                raise ValueError('nodata not found in dataset and product definition')
            self._nodata = num2numpy(self._nodata, self.dtype)
            self._is_2d = len(self.da.dims) == 2
            self.time_idx = self.set_time_idx(time_idx)

        @property
        def nodata(self) -> Optional[float]:
            return self._nodata  # type: ignore

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
            return self.da.shape if self._is_2d else self.da.shape[1:]

        def set_time_idx(self, time_idx: Optional[int]) -> int:
            """
            Updates time index from BandInfo.band

            The resultant time index must be > 0.

            :param int time_index: The time index from BandInfo.band
            :return: The updated time index
            """
            self.time_idx = time_idx or 1
            # adjust for 0 based indexing
            self.time_idx -= 1

            time_count = 1 if self._is_2d else self.da[self.da.dims[0]].size
            if time_count == 0:
                raise ValueError('Found 0 time slices in storage')

            if self.time_idx < time_count:
                return self.time_idx
            else:
                raise ValueError(f'time_idx exceeded {time_count}')

        def read(
            self,
            window: Optional[RasterWindow] = None,
            out_shape: Optional[RasterShape] = None,
        ) -> np.ndarray:
            """
            Reads a slice into the xr.DataArray.

            :param RasterWindow window: The slice to read
            :param RasterShape out_shape: The desired output shape
            :return: Requested data in a :class:`numpy.ndarray`
            """

            # Check if zarr dataset is a 2D array
            t_ix: Tuple = tuple() if self._is_2d else (self.time_idx,)

            if window is None:
                xy_ix: Tuple = (...,)
            else:
                xy_ix = tuple(slice(*w) for w in window)

            data = self.da.values[t_ix + xy_ix]
            return data

    def __init__(self, band: BandInfo):
        """
        Initialises the ZarrDataSource class.

        :param BandInfo band: BandInfo containing the dataset metadata.
        """
        self._band_info = band
        if band.band == 0:
            raise ValueError('BandInfo.band must be > 0')

        # convert band.uri -> protocol, root and group
        protocol, _, _ = uri_split(band.uri)
        if protocol not in PROTOCOL + ['zarr']:
            raise ValueError('Expected file:// or zarr:// url')

        self.uri = band.uri

    @contextmanager
    def open(self) -> Generator[BandDataSource, None, None]:
        """
        Opens a Zarr endpoint.
        This only loads metadata, in preperations for reads.
        """
        zio = ZarrIO()
        ds = zio.open_dataset(uri=self.uri)
        var_name = self._band_info.layer or self._band_info.name
        yield ZarrDataSource.BandDataSource(
            dataset=ds,
            var_name=var_name,
            time_idx=self._band_info.band,
            no_data=self._band_info.nodata,
        )


class ZarrReaderDriver(object):
    def __init__(self) -> None:
        self.name = 'ZarrReader'
        self.protocols = PROTOCOL + ['zarr']
        self.formats = [FORMAT]

    def supports(self, protocol: str, fmt: str) -> bool:
        return protocol in self.protocols and fmt in self.formats

    def new_datasource(self, band: BandInfo) -> ZarrDataSource:
        return ZarrDataSource(band)


def reader_driver_init() -> ZarrReaderDriver:
    return ZarrReaderDriver()


class ZarrWriterDriver(object):
    def __init__(self) -> None:
        pass

    @property
    def aliases(self) -> List:
        return [f'{a} {b}' for a, b in itertools.product([FORMAT], PROTOCOL)]

    @property
    def format(self) -> str:
        return FORMAT

    def mk_uri(self, file_path: str, storage_config: dict) -> str:
        """
        Constructs a URI from the file_path and storage config.

        :param Path file_path: The file path of the file to be converted into a URI.
        :param dict storage_config: The dict holding the storage config found in the
                                    ingest definition.
        :return: file_path as a URI that the Driver understands.
        """
        driver_alias = storage_config['driver']
        if driver_alias not in self.aliases:
            raise ValueError(f'Unknown driver alias: {driver_alias}')

        protocol = driver_alias.split()[1]
        uri = uri_join(protocol, *str(file_path).split("#", 1))
        return uri

    def write_dataset_to_storage(
        self,
        dataset: xr.Dataset,
        file_uri: str,
        global_attributes: Optional[dict] = None,
        variable_params: Optional[dict] = None,
        storage_config: Optional[dict] = None,
        **kwargs: str,
    ) -> Dict:
        """
        Persists a xr.DataSet to storage.

        :param xr.Dataset dataset: The xarray.Dataset to be saved
        :param str file_uri: The file uri
        :param Dict global_attributes: Global attributes from the product definition.
        :param Dict variable_params: Variable parameters from the product definition
        :param Dict storage_config: Storage config from the product definition
        :return: a dict containing additional metadata to be saved in the DB
        """
        # Flattening atributes: Zarr doesn't allow dicts
        for var_name in dataset.data_vars:
            data_var = dataset.data_vars[var_name]
            if 'spectral_definition' in data_var.attrs:
                spectral_definition = data_var.attrs.pop('spectral_definition', None)
                data_var.attrs['dc_spectral_definition_response'] = spectral_definition[
                    'response'
                ]
                data_var.attrs['dc_spectral_definition_wavelength'] = spectral_definition[
                    'wavelength'
                ]

        # Renaming units: units is a reserved name in Xarray coordinates
        for var_name in dataset.coords:
            coord_var = dataset.coords[var_name]
            if 'units' in coord_var.attrs:
                units = coord_var.attrs.pop('units', None)
                coord_var.attrs['dc_units'] = units

        zio = ZarrIO()
        # Should be a directory but actually get passed a file, which becomes a directory.
        metadata = zio.save_dataset_to_zarr(
            uri=file_uri,
            dataset=dataset,
            global_attributes=global_attributes,
            variable_params=variable_params,
            storage_config=storage_config,
        )

        # extra metadata to be stored in database
        return metadata


def writer_driver_init() -> ZarrWriterDriver:
    return ZarrWriterDriver()
