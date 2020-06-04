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

from .utils.uris import uri_split


class ZarrBase():
    def __init__(self) -> None:
        """
        :param str protocol: Supported protocols are ['s3', 'file']
        """

        self._logger = logging.getLogger(self.__class__.__name__)


class ZarrIO(ZarrBase):
    """
    Zarr read/write interface to save and load xarray.Datasets and xarray DataArrays.

    Storage support: [S3, Disk]

    Future:
      - Sparse tree of dense arrays.
      - Parallel IO (immediate, delayed)
        - Local:
          - ThreadPoolExecutor
        - External:
          - dask.distributed with dc.load e.g. with `dask_chunks={'time': 1})`
    Example usage:
        Saving a xarray.Dataset in S3:
            root = 'easi-dc-data/staging/zarr-peter/store'
            data = xr.DataArray(np.random.randn(1300, 1300))

            zio = ZarrIO(protocol='s3')
            # Clean storage area
            zio.clean_store(root=root)
            # Persist to s3
            zio.save_dataset(root=root,
                             group_name='dataset1',
                             dataset=data.to_dataset(name='array1'),
                             chunks={'dim_0': 1100, 'dim_1': 1100})

        Saving a xarray.Dataset on disk:
            root = '/home/ubuntu/odc/test/data'
            data = xr.DataArray(np.random.randn(1300, 1300))

            zio = ZarrIO(protocol='file')
            # Clean storage area
            zio.clean_store(root=root)
            # Persist to file
            zio.save_dataset(root=root,
                             group_name='dataset1',
                             dataset=data.to_dataset(name='array1'),
                             chunks={'dim_0': 1100, 'dim_1': 1100})

        Loading a xarray.Dataset:
            # Open descriptor
            ds1 = zio.open_dataset(root=root, group_name='dataset1')
            # Load data into memory
            ds2 = zio.load_dataset(root=root, group_name='dataset1')
    """

    # Allowed Zarr write modes.
    WRITE_MODES = ('w', 'w-', 'a')

    def __init__(self) -> None:

        super().__init__()

    def print_tree(self,
                   uri: str) -> zarr.util.TreeViewer:
        """
        Prints the Zarr array tree structure.

        :param str uri: The storage uri.
        """
        store = self.get_root(uri)
        group = zarr.open_group(store=store, mode="r")
        return group.tree()

    def get_root(
        self, uri: str
    ) -> Union[fsspec.mapping.FSMap, zarr.storage.DirectoryStore]:
        """
        Sets up the Zarr Group root for IO operations, similar to a Path.

        :param str uri: The storage uri.
        """
        protocol, root, group = uri_split(uri)
        if protocol == 's3':
            store = s3fs.S3Map(root=root,
                               s3=s3fs.S3FileSystem(client_kwargs=dict(region_name=auto_find_region())),
                               check=False)
        elif protocol == 'file':
            store = zarr.DirectoryStore(root)
        else:
            raise ValueError(f'unknown protocol: {protocol}')

        return store

    def clean_store(self,
                    uri: str) -> None:
        """
        Cleans the Zarr store.
        Will delete everything from the group root and below.

        :param str uri: The storage uri.
        """
        protocol, root, group_name = uri_split(uri)
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

    def save_dataarray(self,
                       uri: str,
                       dataarray: xr.DataArray,
                       name: str,
                       chunks: Optional[dict] = None,
                       mode: str = 'w-') -> None:
        """
        Saves a xarray.DataArray

        :param str root: The storage root path.
        :param str group_name: The name of the group
        :param `xarray.DataArray` dataarray: The xarray.DataArray to be saved
        :param str name: The name of the xarray.DataArray
        :param dict chunks: The chunking parameter for each dimension.
        :param str mode: {'w', 'w-', 'a', None}
            w: overwrite
            w-: overwrite if exists
            a: overwrite existing variables (create if does not exist)
        """
        dataset = dataarray.to_dataset(name=name)
        self.save_dataset(
            uri=uri, dataset=dataset,
            chunks=chunks, mode=mode
        )

    def save_dataset(self,
                     uri: str,
                     dataset: xr.Dataset,
                     chunks: Optional[dict] = None,
                     mode: str = 'w-') -> None:
        """
        Saves a xarray.Dataset

        :param str uri: The storage uri.
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

        protocol, root, group = uri_split(uri)
        store = self.get_root(uri)
        dataset.to_zarr(store=store,
                        group=group,
                        mode=mode,
                        consolidated=True,
                        encoding={var: {'compressor': compressor} for var in dataset.data_vars})

    def open_dataset(self,
                     uri: str) -> xr.Dataset:
        """
        Opens a xarray.Dataset

        :param str uri: The storage uri.
        :param str group_name: The group_name to store under root
        """
        protocol, root, group = uri_split(uri)
        store = self.get_root(uri)
        ds: xr.Dataset = xr.open_zarr(store=store, group=group, consolidated=True)
        return ds

    def load_dataset(self,
                     uri: str) -> xr.Dataset:
        """
        Loads a xarray.Dataset

        :param str root: The storage root path.
        :param str group_name: The group_name to store under root
        """
        ds: xr.Dataset = self.open_dataset(uri)
        ds.load()
        return ds

    def save_dataset_to_zarr(self,
                             uri: str,
                             dataset: xr.Dataset,
                             global_attributes: Optional[dict] = None,
                             variable_params: Optional[dict] = None,
                             storage_config: Optional[dict] = None) -> Dict[str, Any]:
        """
        ODC driver calls this
        Saves a Data Cube style xarray Dataset to a Storage Unit

        Requires a spatial Dataset, with attached coordinates and global crs attribute.

        :param str uri: The storage uri.
        :param `xarray.Dataset` dataset:
        :param group: The group name for the dataset
        :param global_attributes: Global file attributes. dict of attr_name: attr_value
        :param variable_params: dict of variable_name: {param_name: param_value, [...]}
                                Allows setting storage and compression options per variable.
        """
        chunks = None
        if storage_config:
            chunks = storage_config['chunking']

        metadata: Dict[str, Any] = {}
        self.save_dataset(uri=uri,
                          dataset=dataset,
                          chunks=chunks)
        return metadata
