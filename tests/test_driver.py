from datacube.drivers import reader_drivers, writer_drivers


def test_reader_drivers():
    available_drivers = reader_drivers()
    assert isinstance(available_drivers, list)
    assert 'zarr' in available_drivers


def test_writer_drivers():
    available_drivers = writer_drivers()
    for name in ('zarr_file', 'zarr_s3', 'zarr file', 'zarr s3'):
        assert name in available_drivers


def test_zarr_netcdf_driver_import():
    try:
        import zarr_io.driver
    except ImportError:
        assert False and 'Failed to load zarr driver'

    assert zarr_io.driver.reader_driver_init is not None
