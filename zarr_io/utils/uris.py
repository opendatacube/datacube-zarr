from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import urlparse


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


def make_zarr_uri(root: Path, group: Optional[str] = None) -> str:
    """Compose zarr uri from path: <protocol>://<root>[#<group>]."""
    uri = f"{root.as_uri()}" + (f"#{group}" if group else "")
    return uri
