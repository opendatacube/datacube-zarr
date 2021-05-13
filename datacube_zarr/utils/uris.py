from typing import MutableMapping, Optional, Tuple
from urllib.parse import urlparse

import s3fs
import zarr


def uri_split(uri: str) -> Tuple[str, str, str]:
    """
    Splits uri into protocol, root, and group

    Example:
        uri_split('file:///path/to/my_dataset.zarr#group/subgroup/etc')
        returns ('file', '/path/to/my_dataset.zarr', 'group/subgroup/etc')

    If the URI contains no '#' extension, the root group "" is returned.

    :param str uri: The URI to be parsed
    :return: (protocol, root, group)
    """
    components = urlparse(uri)
    scheme = components.scheme
    path = components.netloc + components.path
    if not scheme:
        raise ValueError(f'uri scheme not found: {uri}')
    group = components.fragment
    return scheme, path, group


def uri_join(protocol: str, root: str, group: Optional[str] = None) -> str:
    """Compose zarr uri from components: <protocol>://<root>[#<group>].

    :param protocol: storage protocol ('file' or 's3')
    :param root: location of zarr dataset root
    :param group: name of zarr group
    :return: zarr URI '<protocol>://<root>[#<group>]'
    """
    uri = f"{protocol}://{root}" + (f"#{group}" if group else "")
    return uri


def uri_to_store_and_group(uri: str) -> Tuple[MutableMapping, str]:
    """Convert a '<protocol>://<path>#<group>' sting to a zarr storage class and group."""

    # With new s3fs release we can use zarr.FSStore directly or via `normalize_store_args`
    #
    # Something like:
    #
    # store_uri, group = uri.split("#", 1) if "#" in uri else (uri, '')
    # storage_options = {"normalize_keys": False}
    # store = zarr.creation.normalize_store_arg(
    #    store_uri, clobber=True, storage_options=storage_options
    # )

    protocol, root, group = uri_split(uri)
    if protocol == 's3':
        s3 = s3fs.S3FileSystem()
        s3.invalidate_cache()
        store = s3.get_mapper(root=root, check=False)
    elif protocol == 'file':
        store = zarr.DirectoryStore(root)
    else:
        raise ValueError(f'Protocol not known: {protocol}')

    return store, group
