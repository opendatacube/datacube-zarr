from contextlib import ExitStack, contextmanager
from typing import Iterator, Union

from dask import config


@contextmanager
def dask_threadsafe_config(protocol: str) -> Iterator[Union[config.set, ExitStack]]:
    """
    Creates a single worker Dask context if the `protocol` is `s3`
    and the Dask `scheduler` is `threaded`, otherwise unchanged.

    This is a workaround for a suspected thread-safety issue with s3fs on writes

    :param protocol: The storage URI.
    :return: thread-safe contextmanager
    """
    # Set num_workers to 1 due to a thread-safety issue with moto + s3fs
    if protocol == 's3' and config.config.get('scheduler', None) in [
        'threads',
        None,
    ]:
        yield config.set(num_workers=1)
    else:
        yield ExitStack()
