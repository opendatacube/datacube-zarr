import pytest

from datacube_zarr.utils.retry import retry


def test_retry():
    """Test retry decorator."""

    @retry(on_exceptions=(RuntimeError,))
    def fn():
        if fn.count == 0:
            fn.count += 1
            raise RuntimeError("Fail")
        else:
            fn.count += 1
        return fn.count

    fn.count = 0

    assert fn() > 0


def test_retry_fail():
    """Test retry decorator exception."""

    with pytest.raises(RuntimeError) as excinfo:

        @retry(on_exceptions=(RuntimeError,))
        def fn():
            raise RuntimeError("Fail")

        fn()
    assert str(excinfo.value) == 'Fail'
