import logging
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import fsspec
import numpy as np
import s3fs
import xarray as xr
import zarr
from numcodecs import Zstd

from datacube.utils.aws import auto_find_region


class ZarrBase():
    def __init__(self,
                 protocol: Optional[str] = 's3'):
        """
        :param str protocol: Supported protocols are ['s3', 'file']
        """

        self._logger = logging.getLogger(self.__class__.__name__)

        if protocol not in ['s3', 'file']:
            raise ValueError(f'unknown protocol: {protocol}')

        if protocol not in ['s3', 'file']:
            raise ValueError(f'unknown protocol: {protocol}')

        self.protocol = protocol


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
                             relative=True,
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
                             relative=True,
                             dataset=data.to_dataset(name='array1'),
                             chunks={'dim_0': 1100, 'dim_1': 1100})

        Loading a xarray.Dataset:
            # Open descriptor
            ds1 = zio.open_dataset(root=root, group_name='dataset1', relative=True)
            # Load data into memory
            ds2 = zio.load_dataset(root=root, group_name='dataset1', relative=True)
    """

    def __init__(self,
                 protocol: Optional[str] = 's3'):

        super().__init__(protocol)

    def print_tree(self,
                   root: str) -> zarr.util.TreeViewer:
        """
        Prints the Zarr array tree structure.

        :param str root: The storage root path.
        """
        _, group = self.get_root(root)
        return group.tree()

    def get_root(self,
                 root: str) -> Tuple[Union[fsspec.mapping.FSMap, zarr.storage.DirectoryStore], zarr.hierarchy.Group]:
        """
        Sets up the Zarr Group root for IO operations, similar to a Path.

        :param str root: The storage root path.
        """
        if self.protocol == 's3':
            store = s3fs.S3Map(root=root,
                               s3=s3fs.S3FileSystem(client_kwargs=dict(region_name=auto_find_region())),
                               check=False)
        else:
            store = zarr.DirectoryStore(root)

        group = zarr.group(store=store)
        return store, group

    def new_store(self,
                  store: Union[fsspec.mapping.FSMap, zarr.storage.DirectoryStore],
                  group: zarr.hierarchy.Group,
                  group_name: str
                  ) -> Union[fsspec.mapping.FSMap, zarr.storage.DirectoryStore]:
        """
        Creates a new root store
        """
        if group_name not in group:
            group = group.create_group(group_name)
        if isinstance(store, fsspec.mapping.FSMap):
            group_url = ''.join((group.store.root, group.name))
            self._logger.debug(f'S3 group url: {group_url}')
            store = s3fs.S3Map(group_url, s3=store.fs, check=True)
        else:
            group_url = '/'.join((store.path, group.name))
            self._logger.debug(f'File group url: {group_url}')
            store = zarr.DirectoryStore(group_url)
        return store, group

    def clean_store(self,
                    root: str) -> None:
        """
        Cleans the Zarr store.
        Will delete everything from the group root and below.

        :param str root: The storage root path.
        """
        store, group = self.get_root(root)
        if self.protocol == 's3':
            self._logger.info(f'Deleting S3 {root}')
            group.clear()
            store.clear()
        elif self.protocol == 'file':
            self._logger.info(f'Deleting directory {root}')
            root_path = Path(root)
            if root_path.exists() and root_path.is_dir():
                shutil.rmtree(root_path)

    def save_dataarray(self,
                       root: str,
                       group_name: str,
                       dataarray: xr.DataArray,
                       name: str,
                       chunks: Optional[dict] = None,
                       relative: bool = False) -> None:
        """
        Saves a xarray.DataArray

        :param str root: The storage root path.
        :param str group_name: The name of the group
        :param `xarray.DataArray` dataarray: The xarray.DataArray to be saved
        :param str name: The name of the xarray.DataArray
        :param dict chunks: The chunking parameter for each dimension.
        """
        store, group = self.get_root(root)
        if not relative:
            store, group = self.new_store(store, group, group_name)

        compressor = Zstd(level=9)
        if chunks:
            dataset = dataarray.chunk(chunks).to_dataset(name=name)
        else:
            dataset = dataarray.to_dataset(name=name)
        dataset.to_zarr(store=store,
                        mode='w',
                        consolidated=True,
                        encoding={name: {'compressor': compressor}})

    def save_dataset(self,
                     root: str,
                     group_name: str,
                     dataset: xr.Dataset,
                     chunks: Optional[dict] = None,
                     relative: bool = False) -> None:
        """
        Saves a xarray.Dataset

        :param str root: The storage root path.
        :param str group_name: The name of the group
        :param `xarray.Dataset` dataset: The xarray.Dataset to be saved
        :param dict chunks: The chunking parameter for each dimension.
        """
        store, group = self.get_root(root)
        if not relative and group_name:
            store, group = self.new_store(store, group, group_name)
            group_name = group.name

        compressor = Zstd(level=9)
        if chunks:
            dataset = dataset.chunk(chunks)
        dataset.to_zarr(store=store,
                        group=group_name,
                        mode='w',
                        consolidated=True,
                        encoding={var: {'compressor': compressor} for var in dataset.data_vars})

    def open_dataset(self,
                     root: str,
                     group_name: Optional[str],
                     relative: bool = False) -> xr.Dataset:
        """
        Loads a xarray.Dataset

        :param str root: The storage root path.
        """
        if not relative and group_name:
            root = '/'.join((root, group_name))
        store, group = self.get_root(root)
        ds: xr.Dataset = xr.open_zarr(store=store, group=group_name, consolidated=True)
        return ds

    def load_dataset(self,
                     root: str,
                     group_name: Optional[str],
                     relative: bool = False) -> xr.Dataset:
        """
        Loads a xarray.Dataset

        :param str root: The storage root path.
        """
        ds: xr.Dataset = self.open_dataset(root, group_name=group_name, relative=relative)
        ds.load()
        return ds

    def save_dataset_to_zarr(self,
                             root: str,
                             dataset: xr.Dataset,
                             filename: str,
                             global_attributes: Optional[dict] = None,
                             variable_params: Optional[dict] = None,
                             storage_config: Optional[dict] = None) -> Dict[str, Any]:
        """
        ODC driver calls this
        Saves a Data Cube style xarray Dataset to a Storage Unit

        Requires a spatial Dataset, with attached coordinates and global crs attribute.

        :param str root: The storage root path.
        :param `xarray.Dataset` dataset:
        :param filename: Output filename
        :param global_attributes: Global file attributes. dict of attr_name: attr_value
        :param variable_params: dict of variable_name: {param_name: param_value, [...]}
                                Allows setting storage and compression options per variable.
        """
        chunks = None
        if storage_config:
            chunks = storage_config['chunking']

        metadata: Dict[str, Any] = {}
        self.save_dataset(root=root,
                          group_name=filename,
                          dataset=dataset,
                          chunks=chunks)
        return metadata
