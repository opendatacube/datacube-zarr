from typing import MutableMapping, Optional, Tuple
from urllib.parse import urlparse

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
    """Convert a '<protocol>://<path>#<group>' string to a storage class and group."""
    protocol, root, group = uri_split(uri)
    if protocol == "file":
        store = zarr.DirectoryStore(root)
    else:
        store_uri = uri_join(protocol, root)
        store = zarr.storage.FSStore(store_uri, normalize_keys=False)
    return store, group
