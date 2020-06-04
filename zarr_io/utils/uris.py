from typing import Tuple
from urllib.parse import urlparse


def uri_split(uri: str) -> Tuple[str, str, str]:
    """
    Splits uri into protocol, root, and group

    Example:
        uri_split('file:///path/to/my_dataset.zarr#group')
        returns ('file', '/path/to/my_dataset.zarr', 'group')

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
