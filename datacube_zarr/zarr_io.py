import logging
from typing import Callable, Hashable, Mapping, MutableMapping, Optional, Union

import fsspec
import xarray as xr
import zarr
from datacube.utils.aws import auto_find_region
from numcodecs import Blosc
from xarray.backends.common import ArrayWriter
from xarray.backends.zarr import DIMENSION_KEY, ZarrStore

from .utils.chunk import (
    DEFAULT_COMPRESSION_RATIO,
    ZARR_TARGET_CHUNK_SIZE_MB,
    chunk_dataset,
)
from .utils.retry import retry
from .utils.uris import uri_split, uri_to_store_and_group


class ZarrIO:
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
        self._logger = logging.getLogger(self.__class__.__name__)

        # Set the AWS region if not already set
        fsconf = fsspec.config.conf
        region = fsconf.get("client_kwargs", {}).get("region_name", None)
        if region is None:

            @retry(on_exceptions=(ValueError,))
            def _get_auto_find_region() -> str:
                region: str = auto_find_region()
                return region

            region = _get_auto_find_region()

            if "client_kwargs" not in fsconf:
                fsconf["client_kwargs"] = {}

            fsconf["client_kwargs"].update({"region_name": region})
            self._logger.info(f'Setting AWS region to {region}.')

    def print_tree(self, uri: str) -> zarr.util.TreeViewer:
        """
        Prints the Zarr array tree structure.

        :param str uri: The storage URI.
        :return: A zarr.util.TreeViewer view of the Zarr group
        """
        store = self.get_root(uri)
        group = zarr.open_group(store=store, mode="r")
        return group.tree()

    def get_root(self, uri: str) -> MutableMapping:
        """
        Sets up the Zarr Group root for IO operations, similar to a Path.

        :param str uri: The storage URI.
        :return: The Zarr store for this URI.
        """
        store, _ = uri_to_store_and_group(uri)
        return store

    def clean_store(self, uri: str) -> None:
        """
        Cleans the Zarr store.
        Will delete everything from the group root and below.

        :param str uri: The storage URI.
        """
        store, _ = uri_to_store_and_group(uri)
        store.clear()

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

        dataset = chunk_dataset(dataset, chunks, target_mb, compression_ratio)

        compressor = Blosc(cname='zstd', clevel=4, shuffle=Blosc.BITSHUFFLE)
        encoding = {var: {'compressor': compressor} for var in dataset.data_vars}

        store, group = uri_to_store_and_group(uri)
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
        zarr_args = {"consolidated": True}
        store, group = uri_to_store_and_group(uri)
        ds: xr.Dataset = xr.open_dataset(
            store, group=group, engine="zarr", chunks={}, backend_kwargs=zarr_args
        )
        return ds


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
