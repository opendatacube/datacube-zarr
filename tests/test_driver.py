'''Unit tests for the datacube_zarr.driver module.'''
from pathlib import Path
from random import random, sample

import pytest
import numpy as np
from datacube.drivers import reader_drivers
from datacube.storage import BandInfo
from datacube.testutils.iodriver import open_reader
from mock import MagicMock

from datacube_zarr.driver import (
    ZarrDataSource,
    ZarrReaderDriver,
    reader_driver_init,
    uri_split,
)

SPECTRAL_DEFINITION = {
    'wavelength': sorted(sample(range(380, 750), 150)),
    'response': [random() for i in range(150)],
}
'''Random spectral definition with 150 values.'''


# General ODC Tests
def test_reader_drivers():
    '''Check the zarr reader driver is found by Datacube.'''
    available_drivers = reader_drivers()
    assert isinstance(available_drivers, list)
    assert 'zarr' in available_drivers


def test_zarr_netcdf_driver_import():
    '''Check the zarr modules can be imported.'''
    try:
        import datacube_zarr.driver
    except ImportError:
        assert False and 'Failed to load zarr driver'

    assert datacube_zarr.driver.reader_driver_init is not None


# datacube_zarr.driver Unit tests
@pytest.mark.parametrize('dataset_fixture', ['odc_dataset', 'odc_dataset_2d'])
def test_datasource(request, dataset, dataset_fixture):
    """Test ZarrDataSource.

    Data is saved to file and opened by the data source."""
    odc_dataset_ = request.getfixturevalue(dataset_fixture)

    group_name = list(dataset.keys())[0]
    source = ZarrDataSource(BandInfo(odc_dataset_, group_name))
    with source.open() as band_source:
        ds = band_source.read()
        assert np.array_equal(ds.squeeze(), dataset[group_name].values.squeeze())


def test_datasource_empty_band_info(dataset):
    '''Test ZarrDataSource when the BandInfo has no band.'''
    band_info = MagicMock()
    band_info.band = 0
    with pytest.raises(ValueError) as excinfo:
        ZarrDataSource(band_info)
    assert str(excinfo.value) == 'BandInfo.band must be > 0'


def test_datasource_wrong_protocol(dataset):
    '''Test ZarrDataSource with an invalid protocol.'''
    band_info = MagicMock()
    band_info.band = 1
    band_info.uri = 'foo://bar/baz'
    with pytest.raises(ValueError) as excinfo:
        ZarrDataSource(band_info)
    assert str(excinfo.value) == 'Expected file:// or s3:// url'


def test_datasource_read_window(dataset):
    '''Test the ZarrDataSource.BandDataSource.'''
    group_name = list(dataset.keys())[0]

    band_source = ZarrDataSource.BandDataSource(
        dataset, group_name, dataset[group_name].nodata
    )
    assert band_source.crs == dataset.crs
    assert band_source.transform == dataset[group_name].affine
    assert band_source.dtype == dataset[group_name].dtype
    assert band_source.shape == dataset[group_name].shape[1:]

    ds2 = band_source.read()
    assert np.array_equal(ds2, dataset[group_name].values)

    ds3 = band_source.read((0, (30, 50), (30, 50)))
    assert np.array_equal(ds3, dataset[group_name][0, 30:50, 30:50].values)


def test_datasource_read_outshape(dataset, tmpdir):
    """Test that reads with `out_shape` specified match the rio driver."""
    group_name = list(dataset.keys())[0]
    z_src = ZarrDataSource.BandDataSource(dataset, group_name, dataset[group_name].nodata)

    # Create a ODC `RIOReader` reader from the original test tiff file
    tif = next(Path(tmpdir).iterdir())
    t_src = open_reader(tif)
    for out_shape in (None, (0, 1), (2, 2), (4, 7), (16, 24), (32, 30), (143, 52)):
        t_data = t_src.read(out_shape=out_shape).result()
        z_data = z_src.read((0,), out_shape=out_shape)
        np.testing.assert_array_equal(z_data, t_data)


def test_datasource_bad_window(dataset):
    '''Test the ZarrDataSource.BandDataSource with an invalid time index.'''
    group_name = list(dataset.keys())[0]
    band_source = ZarrDataSource.BandDataSource(
        dataset, group_name, dataset[group_name].nodata
    )

    with pytest.raises(IndexError) as excinfo:
        band_source.read((1,))
    assert str(excinfo.value) == 'index 1 is out of bounds for axis 0 with size 1'


def test_datasource_no_time_slice(dataset):
    '''Test the ZarrDataSource.BandDataSource without time slice.'''
    group_name = list(dataset.keys())[0]
    dataset = dataset.drop_sel(time=dataset.time.values)
    with pytest.raises(ValueError) as excinfo:
        ZarrDataSource.BandDataSource(dataset, group_name, dataset[group_name].nodata)
    assert str(excinfo.value) == 'Dataset has 0 bands.'


def test_datasource_no_nodata(dataset):
    '''Test the ZarrDataSource.BandDataSource without nodata.'''
    group_name = list(dataset.keys())[0]
    dataset[group_name].attrs.pop('nodata', None)
    with pytest.raises(ValueError) as excinfo:
        ZarrDataSource.BandDataSource(dataset, group_name, None)
    assert str(excinfo.value) == 'nodata not found in dataset and product definition'


uri_split_test_params = [
    ('protocol:///some/path/root.zarr', ('protocol', '/some/path/root.zarr', '')),
    (
        's3:///some/path/root.zarr#group/subgroup',
        ('s3', '/some/path/root.zarr', 'group/subgroup'),
    ),
    ('file:///some/path/root.zarr#/', ('file', '/some/path/root.zarr', '/')),
]


def test_datasource_stacked_nodata(dataset):
    '''Test the ZarrDataSource.BandDataSource with nodata as a list.'''
    group_name = list(dataset.keys())[0]
    dataset.aa.attrs['nodata'] = [-9999]
    band_source = ZarrDataSource.BandDataSource(dataset, group_name, None)
    assert band_source.nodata == -9999


@pytest.mark.parametrize("uri,split_uri", uri_split_test_params)
def test_uri_split(uri, split_uri):
    '''Check zarr uri splitting.'''
    assert uri_split(uri) == split_uri


def test_uri_split_no_scheme():
    '''Check error is raised when no scheme present.'''
    with pytest.raises(ValueError) as excinfo:
        uri_split('/some/path/group.zarr')

    assert str(excinfo.value) == 'uri scheme not found: /some/path/group.zarr'


def test_zarr_reader_driver(dataset, odc_dataset):
    '''Check supported protocols and formats.'''
    protocols = ['file', 's3']
    formats = ['zarr']
    reader = reader_driver_init()
    assert isinstance(reader, ZarrReaderDriver)
    assert sorted(reader.protocols) == sorted(protocols)
    assert sorted(reader.formats) == sorted(formats)

    for protocol in protocols:
        for fmt in formats:
            assert reader.supports(protocol, fmt)

    group_name = list(dataset.keys())[0]
    source = reader.new_datasource(BandInfo(odc_dataset, group_name))
    with source.open() as band_source:
        ds = band_source.read()
        assert np.array_equal(ds, dataset[group_name].values)
