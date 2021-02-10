"""
Zarr Storage driver for ODC
Supports storage on S3 and Disk
Should be able to handle hyperspectral data when ready.
"""
import itertools
from contextlib import contextmanager
from json.decoder import JSONDecodeError
from typing import Any, Dict, Generator, List, Optional, Tuple

import numpy as np
import xarray as xr
from affine import Affine
from datacube.storage import BandInfo
from datacube.utils import geometry
from datacube.utils.math import num2numpy

from .utils.retry import retry
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
            no_data: Optional[float],
        ):
            """
            Initialises the BandDataSource class.

            The BandDataSource class to read array slices out of the xr.Dataset.

            :param xr.Dataset dataset: The xr.Dataset
            :param str var_name: The variable name of the xr.DataArray
            :param float no_data: The no data value if known
            """
            self.ds = dataset
            self._var_name = var_name
            self.da = dataset.data_vars[var_name]

            self._is_2d = len(self.da.dims) == 2
            self._nbands = 1 if self._is_2d else self.da[self.da.dims[0]].size
            if self._nbands == 0:
                raise ValueError('Dataset has 0 bands.')

            # Set nodata value
            if 'nodata' in self.da.attrs and self.da.nodata:
                if isinstance(self.da.nodata, list):
                    self._nodata = self.da.nodata[0]
                else:
                    self._nodata = self.da.nodata
            else:
                self._nodata = no_data

            if not self._nodata:
                raise ValueError('nodata not found in dataset and product definition')

            self._nodata = num2numpy(self._nodata, self.dtype)

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

            if window is None:
                ix: Tuple = (...,)
            else:
                ix = tuple(slice(*w) if isinstance(w, tuple) else w for w in window)

            # Fixes intermittent Zarr decompression errors when used with Dask
            # e.g. RuntimeError: error during blosc decompression: 0
            @retry(on_exceptions=(RuntimeError, JSONDecodeError))
            def fn() -> Any:
                return self.da[ix].values

            data = fn()
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
        if protocol not in PROTOCOL:
            raise ValueError('Expected file:// or s3:// url')

        self.uri = band.uri

    @contextmanager
    def open(self) -> Generator[BandDataSource, None, None]:
        """
        Opens a Zarr endpoint.
        This only loads metadata, in preperations for reads.
        """
        zio = ZarrIO()

        # Fixes intermittent Zarr decompression errors when used with Dask
        # e.g. RuntimeError: error during blosc decompression: 0
        @retry(on_exceptions=(RuntimeError, JSONDecodeError))
        def fn() -> Any:
            return zio.open_dataset(uri=self.uri)

        dataset = fn()

        var_name = self._band_info.layer or self._band_info.name
        yield ZarrDataSource.BandDataSource(
            dataset=dataset,
            var_name=var_name,
            no_data=self._band_info.nodata,
        )


class ZarrReaderDriver(object):
    def __init__(self) -> None:
        self.name = 'ZarrReader'
        self.protocols = PROTOCOL
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
