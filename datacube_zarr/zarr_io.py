import logging
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Union

import fsspec
import s3fs
import xarray as xr
import zarr
from datacube.utils.aws import auto_find_region
from numcodecs import Zstd

from datacube_zarr.utils.chunk import auto_chunk_dataset

from .utils.uris import uri_split

_APPROX_ZSTD_COMPRESSION_RATIO = 3.0


class ZarrBase:
    def __init__(self) -> None:
        self._logger = logging.getLogger(self.__class__.__name__)


class ZarrIO(ZarrBase):
    """
    Zarr read/write interface to save and load xarray.Datasets and xarray DataArrays.

    Storage support: [S3, Disk]

    Example usage:
        Saving a xarray.Dataset:
            data = xr.DataArray(np.random.randn(1300, 1300))

            # uri = 'file:///root/mydata.zarr#dataset1'
            uri = 's3://my-bucket/mydata.zarr#dataset1'
            zio = ZarrIO()
            zio.save_dataset(
                uri=uri,
                dataset=data.to_dataset(name='array1'),
                chunks={'dim_0': 1100, 'dim_1': 1100},
            )

        Loading a xarray.Dataset:
            ds = zio.load_dataset(uri=uri)
    """

    # Allowed Zarr write modes.
    WRITE_MODES = ('w', 'w-', 'a')

    def __init__(self) -> None:

        super().__init__()

    def print_tree(self, uri: str) -> zarr.util.TreeViewer:
        """
        Prints the Zarr array tree structure.

        :param str uri: The storage URI.
        :return: A zarr.util.TreeViewer view of the Zarr group
        """
        store = self.get_root(uri)
        group = zarr.open_group(store=store, mode="r")
        return group.tree()

    def get_root(
        self, uri: str
    ) -> Union[fsspec.mapping.FSMap, zarr.storage.DirectoryStore]:
        """
        Sets up the Zarr Group root for IO operations, similar to a Path.

        :param str uri: The storage URI.
        :return: The Zarr store for this URI.
        """
        protocol, root, _ = uri_split(uri)
        if protocol == 's3':
            store = s3fs.S3Map(
                root=root,
                s3=s3fs.S3FileSystem(
                    client_kwargs=dict(region_name=auto_find_region()),
                    use_listings_cache=False,
                ),
                check=False,
            )
        elif protocol == 'file':
            store = zarr.DirectoryStore(root)
        else:
            raise ValueError(f'unknown protocol: {protocol}')

        return store

    def clean_store(self, uri: str) -> None:
        """
        Cleans the Zarr store.
        Will delete everything from the group root and below.

        :param str uri: The storage URI.
        """
        protocol, root, _ = uri_split(uri)
        if protocol == 's3':
            self._logger.info(f'Deleting S3 {root}')
            store = self.get_root(uri)
            group = zarr.group(store=store)
            group.clear()
            store.clear()
        elif protocol == 'file':
            self._logger.info(f'Deleting directory {root}')
            root_path = Path(root)
            if root_path.exists() and root_path.is_dir():
                shutil.rmtree(root_path)

    def save_dataarray(
        self,
        uri: str,
        dataarray: xr.DataArray,
        name: str,
        chunks: Optional[dict] = None,
        mode: str = 'w-',
    ) -> None:
        """
        Saves a xarray.DataArray

        :param str uri: The output URI.
        :param `xarray.DataArray` dataarray: The xarray.DataArray to be saved
        :param str name: The name of the xarray.DataArray
        :param dict chunks: The chunking parameter for each dimension.
        :param str mode: {'w', 'w-', 'a', None}
            w: overwrite
            w-: overwrite if exists
            a: overwrite existing variables (create if does not exist)
        """
        dataset = dataarray.to_dataset(name=name)
        self.save_dataset(uri=uri, dataset=dataset, chunks=chunks, mode=mode)

    def save_dataset(
        self,
        uri: str,
        dataset: xr.Dataset,
        chunks: Optional[dict] = None,
        mode: str = 'w-',
    ) -> None:
        """
        Saves a xarray.Dataset

        :param str uri: The output storage URI.
        :param `xarray.Dataset` dataset: The xarray.Dataset to be saved
        :param dict chunks: The chunking parameter for each dimension.
        :param str mode: {'w', 'w-', 'a', None}
            w: overwrite
            w-: overwrite if exists
            a: overwrite existing variables (create if does not exist)
        """
        if mode not in self.WRITE_MODES:
            raise ValueError(f"Only the following modes are supported {self.WRITE_MODES}")

        compressor = Zstd(level=9)

        if chunks:
            dataset = dataset.chunk(chunks)
        else:
            dataset = auto_chunk_dataset(
                ds=dataset,
                target_mb=20,
                compressor=compressor,
                default_compression_ratio=_APPROX_ZSTD_COMPRESSION_RATIO,
            )

        encoding = {var: {'compressor': compressor} for var in dataset.data_vars}

        _, _, group = uri_split(uri)
        store = self.get_root(uri)
        dataset.to_zarr(
            store=store, group=group, mode=mode, consolidated=True, encoding=encoding,
        )

    def open_dataset(self, uri: str) -> xr.Dataset:
        """
        Opens a xarray.Dataset

        :param str uri: The storage URI.
        :param str group_name: The group_name to store under root
        :return: The opened xr.Dataset
        """
        _, _, group = uri_split(uri)
        store = self.get_root(uri)
        ds: xr.Dataset = xr.open_zarr(store=store, group=group, consolidated=True)
        return ds

    def load_dataset(self, uri: str) -> xr.Dataset:
        """
        Loads a xarray.Dataset

        :param str uri: The dataset URI
        :return: The loaded xr.Dataset
        """
        ds: xr.Dataset = self.open_dataset(uri)
        ds.load()
        return ds

    def save_dataset_to_zarr(
        self,
        uri: str,
        dataset: xr.Dataset,
        global_attributes: Optional[dict] = None,
        variable_params: Optional[dict] = None,
        storage_config: Optional[dict] = None,
    ) -> Dict[str, Any]:
        """
        ODC driver calls this
        Saves a Data Cube style xarray Dataset to a Storage Unit

        Requires a spatial Dataset, with attached coordinates and global crs attribute.

        :param str uri: The output storage URI.
        :param `xarray.Dataset` dataset: The xarray Dataset to be saved to Zarr
        :param dict global_attributes: Global file attributes.
                                       dict of attr_name: attr_value
        :param dict variable_params: dict of variable_name:
                                       {param_name: param_value, [...]}
                                     Allows setting storage and compression options per
                                     variable.
        :param dict storage_config: The storage config from the ingest definition.
        :return: dict containing additional driver metadata to be stored in the database
        """
        chunks = None
        if storage_config:
            chunks = storage_config['chunking']

        metadata: Dict[str, Any] = {}
        self.save_dataset(uri=uri, dataset=dataset, chunks=chunks)
        return metadata
