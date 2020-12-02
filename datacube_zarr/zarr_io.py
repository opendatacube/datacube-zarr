import logging
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, Hashable, Mapping, Optional, Union

import fsspec
import s3fs
import xarray as xr
import zarr
from datacube.utils.aws import auto_find_region
from numcodecs import Zstd
from xarray.backends.common import ArrayWriter
from xarray.backends.zarr import DIMENSION_KEY, ZarrStore

from .utils.chunk import (
    DEFAULT_COMPRESSION_RATIO,
    ZARR_TARGET_CHUNK_SIZE_MB,
    chunk_dataset,
)
from .utils.context_manager import dask_threadsafe_config
from .utils.uris import uri_split


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
        chunks: Optional[Mapping[Hashable, Union[str, int]]] = None,
        mode: str = 'w-',
        target_mb: float = ZARR_TARGET_CHUNK_SIZE_MB,
        compression_ratio: float = DEFAULT_COMPRESSION_RATIO,
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
        dataset = chunk_dataset(dataset, chunks, target_mb, compression_ratio)
        encoding = {var: {'compressor': compressor} for var in dataset.data_vars}

        protocol, _, group = uri_split(uri)
        store = self.get_root(uri)

        with dask_threadsafe_config(protocol):
            dataset.to_zarr(
                store=store, group=group, mode=mode, consolidated=True, encoding=encoding
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


def _xarray_dim_rename_visitor(
    old: str, new: str
) -> Callable[[Union[zarr.Array, zarr.Group]], None]:
    """Change xarray attributes specifying dataset dims."""

    def update_xr_dimension_key(zval: Union[zarr.Array, zarr.Group]) -> None:
        """Replace xarray dimension name for each array."""
        zval.attrs[DIMENSION_KEY] = [
            new if k == old else k for k in zval.attrs[DIMENSION_KEY]
        ]

    return update_xr_dimension_key


def replace_dataset_dim(uri: str, dim: str, new: Union[str, xr.IndexVariable]) -> None:
    """Replace a dataset dimension with a new name or entire coordinates.

    :param uri: The dataset URI
    :param dim: Name of the dimension to rename/replace
    :param new: The new dimension name or named 1D coordinate data
    """
    root = ZarrIO().get_root(uri)
    _, _, group = uri_split(uri)
    is_consolidated = ".zmetadata" in root
    zstore = ZarrStore.open_group(
        root, mode="r+", group=group, consolidate_on_close=is_consolidated
    )

    new_name = new if isinstance(new, str) else new.name

    if dim not in zstore.get_dimensions():
        raise KeyError(f"Dimension '{dim}' does not exist.")

    if new_name in zstore.get_variables():
        raise KeyError(f"Dataset already contains variable named '{new_name}'.")

    dim_has_coords = dim in zstore.get_variables()

    if isinstance(new, str):
        if dim_has_coords:
            zstore.ds.move(dim, new_name)
    else:
        if not dim_has_coords:
            raise ValueError(f"Dimension '{dim}' has no coordinates.")

        zarray = zstore.ds[dim]

        if not len(zarray) == len(new):
            raise ValueError(
                f"Data has incompatible length ({len(new)}) with "
                f"dimension '{dim}' ({len(zarray)})."
            )

        if zarray.dtype == new.dtype:
            # If coord data is same dtype, move and assign data in place with zarr
            zarray[:] = new.data
            zstore.ds.move(dim, new.name)
        else:
            # If coord data is a new dtype, delete and add new variable using xarray
            del zstore.ds[dim]
            writer = ArrayWriter()
            zstore.set_variables(
                variables={new.name: new}, check_encoding_set=[], writer=writer
            )
            writer.sync()

    # Update references to the old dimension in xarray attributes
    zstore.ds.visitvalues(_xarray_dim_rename_visitor(dim, new_name))

    # xarray 0.16.2 doesn't really consolidate on close
    if is_consolidated:
        zarr.consolidate_metadata(zstore.ds.store)

    zstore.close()
