from functools import wraps
from typing import Any, Callable, Tuple, Type, TypeVar, Union, cast

F = TypeVar('F', bound=Callable[..., Any])


def retry(
    on_exceptions: Union[Type[Exception], Tuple[Type[Exception], ...]] = Exception,
    max_retries: int = 10,
) -> Callable[[F], F]:
    """
    Retry decorator to retry a function on Exception.

    :param tuple on_exceptions: The exceptions to retry on
    :param int max_retries: The number of retries to attempt.
                            Default is 10.
    :return: The decorated function
    """

    def retry_decorator(f: F) -> F:
        @wraps(f)
        def with_retries(*args: Any, **kwargs: Any) -> Any:
            num_retries = 0
            while True:
                try:
                    return f(*args, **kwargs)
                except on_exceptions:
                    if num_retries > max_retries:
                        raise
                    num_retries += 1

        return cast(F, with_retries)

    return retry_decorator
