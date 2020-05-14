'''Unit tests for the zarr_io.driver module.'''
from pathlib import Path
from random import random, sample

import pytest
import numpy as np
from mock import MagicMock

from datacube.drivers import reader_drivers, writer_drivers
from datacube.storage import BandInfo
from zarr_io.driver import (
    ZarrDataSource,
    ZarrReaderDriver,
    ZarrWriterDriver,
    file_writer_driver_init,
    reader_driver_init,
    s3_writer_driver_init,
    uri_split,
)

from .utils import _check_zarr_files, _load_dataset

SPECTRAL_DEFINITION = {
    'wavelength': sorted(sample(range(380, 750), 150)),
    'response': [random() for i in range(150)]
}
'''Random spectral definition with 150 values.'''


# General ODC Tests
def test_reader_drivers():
    '''Check the zarr reader driver is found by Datacube.'''
    available_drivers = reader_drivers()
    assert isinstance(available_drivers, list)
    assert 'zarr' in available_drivers


def test_writer_drivers():
    '''Check the zarr writer driver is found by Datacube.'''
    available_drivers = writer_drivers()
    for name in ('zarr_file', 'zarr_s3', 'zarr file', 'zarr s3'):
        assert name in available_drivers


def test_zarr_netcdf_driver_import():
    '''Check the zarr modules can be imported.'''
    try:
        import zarr_io.driver
    except ImportError:
        assert False and 'Failed to load zarr driver'

    assert zarr_io.driver.reader_driver_init is not None


# zarr_io.driver Unit tests
@pytest.mark.parametrize('dataset_fixture', ['odc_dataset', 'odc_dataset_2d'])
def test_datasource(request, dataset, dataset_fixture):
    '''Test ZarrDataSource.

    Data is saved to file and opened by the data source.'''
    odc_dataset_ = request.getfixturevalue(dataset_fixture)
    group_name = list(dataset.keys())[0]
    source = ZarrDataSource(BandInfo(odc_dataset_, group_name))
    with source.open() as band_source:
        ds = band_source.read()
        assert np.array_equal(ds, dataset[group_name].values[0, ...])


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
    assert str(excinfo.value) == 'Expected file:// or zarr:// url'


def test_datasource_no_timeslice(dataset):
    '''Test the ZarrDataSource.BandDataSource.'''
    group_name = list(dataset.keys())[0]

    band_source = ZarrDataSource.BandDataSource(dataset, group_name, None)
    assert band_source.crs == dataset.crs
    assert band_source.transform == dataset[group_name].affine
    assert band_source.dtype == dataset[group_name].dtype
    assert band_source.shape == dataset[group_name].shape[1:]

    ds2 = band_source.read()
    assert np.array_equal(ds2, dataset[group_name].values[0, ...])

    ds3 = band_source.read(((30, 50), (30, 50)))
    assert np.array_equal(ds3, dataset[group_name][0, 30:50, 30:50].values)


def test_datasource_bad_time_index(dataset):
    '''Test the ZarrDataSource.BandDataSource with an invalid time index.'''
    group_name = list(dataset.keys())[0]
    with pytest.raises(ValueError) as excinfo:
        ZarrDataSource.BandDataSource(dataset, group_name, dataset.time.size + 1)
    assert str(excinfo.value) == 'time_idx exceeded 1'


def test_datasource_no_time_slice(dataset):
    '''Test the ZarrDataSource.BandDataSource without time slice.'''
    group_name = list(dataset.keys())[0]
    dataset = dataset.drop_sel(time=dataset.time.values)
    with pytest.raises(ValueError) as excinfo:
        ZarrDataSource.BandDataSource(dataset, group_name, None)
    assert str(excinfo.value) == 'Found 0 time slices in storage'


def test_uri_split():
    '''Check zarr uri splitting.'''
    assert uri_split('protocol:///some/path/group') == ('protocol', '/some/path', 'group')
    assert uri_split('/some/path/group') == ('file', '/some/path/group', None)


def test_zarr_reader_driver(dataset, odc_dataset):
    '''Check supported protocols and formats.'''
    protocols = ['file', 's3', 'zarr']
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
        assert np.array_equal(ds, dataset[group_name].values[0, ...])


def test_zarr_file_writer_driver():
    '''Check aliases, format and uri_scheme for the `file` writer.'''
    writer = file_writer_driver_init()
    assert isinstance(writer, ZarrWriterDriver)
    assert writer.aliases == ['zarr file']
    assert writer.format == 'zarr'
    assert writer.uri_scheme == 'file'


def test_zarr_s3_writer_driver():
    '''Check aliases, format and uri_scheme for the `s3` writer.'''
    writer = s3_writer_driver_init()
    assert isinstance(writer, ZarrWriterDriver)
    assert writer.aliases == ['zarr s3']
    assert writer.format == 'zarr'
    assert writer.uri_scheme == 's3'


def test_zarr_other_writer_driver():
    '''Testing a loophole where the protocol gets updated late.'''
    writer = file_writer_driver_init()
    writer.zio.protocol = 'xxx'
    assert writer.aliases == []


def test_invalid_protocol():
    '''Test exceptions when an invalid protocol is used.'''
    with pytest.raises(ValueError) as excinfo:
        ZarrWriterDriver(protocol='xxx')
    assert str(excinfo.value) == 'unknown protocol: xxx'


@pytest.mark.parametrize('protocol', ('file', 's3'))
def test_zarr_file_writer_driver_save(protocol, fixed_chunks, data, tmpdir, s3):
    '''Test the `write_dataset_to_storage` method.'''
    # write_dataset_to_storage calls save_dataset which uses relative=False by default
    relative = False
    root = s3['root'] if protocol == 's3' \
        else Path(tmpdir) / 'data'
    group_name = 'dataset1'
    name = 'array1'
    writer = ZarrWriterDriver(protocol=protocol)
    ds_in = data.to_dataset(name=name)
    writer.write_dataset_to_storage(
        dataset=ds_in.copy(),
        filename=f'{root}/{group_name}',
        storage_config={'chunking': fixed_chunks['input']}
    )
    if protocol == 'file':
        _check_zarr_files(data, root, group_name, name, relative, fixed_chunks)
    # Load and check data
    ds_out = _load_dataset(protocol, root, group_name, relative=relative)
    assert ds_in.equals(ds_out)  # Compare values only


def test_zarr_file_writer_driver_data_corrections(fixed_chunks, data, tmpdir):
    '''Test dataset key corrections applied by `write_dataset_to_storage`.'''
    # write_dataset_to_storage calls save_dataset which uses relative=False by default
    relative = False
    protocol = 'file'
    root = Path(tmpdir) / 'data'
    group_name = 'dataset1'
    name = 'array1'
    writer = ZarrWriterDriver(protocol=protocol)
    ds_in = data.to_dataset(name=name)
    # Assign target keys: spectral definition and coords attributes
    ds_in.array1.attrs['spectral_definition'] = SPECTRAL_DEFINITION
    coords = {dim: [1] * size for dim, size in ds_in.dims.items()}
    ds_in = ds_in.assign_coords(coords)
    for coord_name in ds_in.coords:
        ds_in.coords[coord_name].attrs['units'] = 'Fake unit'
    writer.write_dataset_to_storage(
        dataset=ds_in.copy(),  # The copy should be corrected
        filename=f'{root}/{group_name}',
        storage_config={'chunking': fixed_chunks['input']}
    )
    # Load and check data has been corrected
    ds_out = _load_dataset(protocol, root, group_name, relative=relative)
    assert ds_in.equals(ds_out)  # Values only
    for key, value in SPECTRAL_DEFINITION.items():
        # spectral defs attributes should now start with 'dc_'
        assert ds_out.array1.attrs[f'dc_spectral_definition_{key}'] == value  # attrs
    for coord_name in ds_in.coords:
        assert ds_out.coords[coord_name].equals(ds_in.coords[coord_name])
        for attr, val in ds_in.coords[coord_name].attrs.items():
            # units attribute should now start with 'dc_'
            if attr == 'units':
                attr = 'dc_units'
            assert ds_out.coords[coord_name].attrs[attr] == val
