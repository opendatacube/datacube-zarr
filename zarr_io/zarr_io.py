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
    def __init__(self,
                 protocol: str = 's3'):
        """
        :param str protocol: Supported protocols are ['s3', 'file']
        """

        self._logger = logging.getLogger(self.__class__.__name__)

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

    # Allowed Zarr write modes.
    WRITE_MODES = ('w', 'w-', 'a')

    def __init__(self,
                 protocol: str = 's3'):

        super().__init__(protocol)

    def print_tree(self,
                   root: str) -> zarr.util.TreeViewer:
        """
        Prints the Zarr array tree structure.

        :param str root: The storage root path.
        """
        store = self.get_root(root)
        group = zarr.open_group(store=store, mode="r")
        return group.tree()

    def get_root(
        self, root: str
    ) -> Union[fsspec.mapping.FSMap, zarr.storage.DirectoryStore]:
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

        return store

    def new_store(self,
                  store: Union[fsspec.mapping.FSMap, zarr.storage.DirectoryStore],
                  group: zarr.hierarchy.Group,
                  group_name: str
                  ) -> Union[fsspec.mapping.FSMap, zarr.storage.DirectoryStore]:
        """
        Creates a new root store.
        """
        if group_name not in group:
            group = group.create_group(group_name)
        else:
            group = group[group_name]
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
        if self.protocol == 's3':
            self._logger.info(f'Deleting S3 {root}')
            store = self.get_root(root)
            group = zarr.group(store=store)
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
                       mode: str = 'w-',
                       relative: bool = True) -> None:
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
        :param bool relative: True for relative indexing and False for global indexing
        """
        dataset = dataarray.to_dataset(name=name)
        self.save_dataset(
            root=root, group_name=group_name, dataset=dataset,
            chunks=chunks, mode=mode, relative=relative
        )

    def save_dataset(self,
                     root: str,
                     group_name: str,
                     dataset: xr.Dataset,
                     chunks: Optional[dict] = None,
                     mode: str = 'w-',
                     relative: bool = True) -> None:
        """
        Saves a xarray.Dataset

        :param str root: The storage root path.
        :param str group_name: The name of the group
        :param `xarray.Dataset` dataset: The xarray.Dataset to be saved
        :param dict chunks: The chunking parameter for each dimension.
        :param str mode: {'w', 'w-', 'a', None}
            w: overwrite
            w-: overwrite if exists
            a: overwrite existing variables (create if does not exist)
        :param bool relative: True for relative indexing and False for global indexing
        """
        if mode not in self.WRITE_MODES:
            raise ValueError(f"Only the following modes are supported {self.WRITE_MODES}")

        store = self.get_root(root)
        if not relative and group_name:
            group = zarr.group(store=store)
            store, group = self.new_store(store, group, group_name)
            group_name = group.name

        compressor = Zstd(level=9)
        if chunks:
            dataset = dataset.chunk(chunks)
        dataset.to_zarr(store=store,
                        group=group_name,
                        mode=mode,
                        consolidated=True,
                        encoding={var: {'compressor': compressor} for var in dataset.data_vars})

    def open_dataset(self,
                     root: str,
                     group_name: Optional[str] = None,
                     relative: bool = True) -> xr.Dataset:
        """
        Opens a xarray.Dataset

        :param str root: The storage root path.
        :param str group_name: The group_name to store under root
        :param bool relative: Whether to use relative or absolute addressing.
        """
        if not relative and group_name:
            root = '/'.join((root, group_name))
        store = self.get_root(root)
        ds: xr.Dataset = xr.open_zarr(store=store, group=group_name, consolidated=True)
        return ds

    def load_dataset(self,
                     root: str,
                     group_name: Optional[str] = None,
                     relative: bool = True) -> xr.Dataset:
        """
        Loads a xarray.Dataset

        :param str root: The storage root path.
        :param str group_name: The group_name to store under root
        :param bool relative: Whether to use relative or absolute addressing.
        """
        ds: xr.Dataset = self.open_dataset(root, group_name=group_name, relative=relative)
        ds.load()
        return ds

    def save_dataset_to_zarr(self,
                             root: str,
                             dataset: xr.Dataset,
                             group: str,
                             global_attributes: Optional[dict] = None,
                             variable_params: Optional[dict] = None,
                             storage_config: Optional[dict] = None) -> Dict[str, Any]:
        """
        ODC driver calls this
        Saves a Data Cube style xarray Dataset to a Storage Unit

        Requires a spatial Dataset, with attached coordinates and global crs attribute.

        :param str root: The storage root path.
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
        self.save_dataset(root=root,
                          group_name=group,
                          dataset=dataset,
                          chunks=chunks)
        return metadata
