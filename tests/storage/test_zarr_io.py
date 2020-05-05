'''Low-level zarr_io tests.'''
from json import load
from os import environ
from pathlib import Path
from types import SimpleNamespace

import pytest
from mock import MagicMock
import boto3
import mock
import numpy as np
from moto import mock_s3
from xarray import DataArray

from datacube import Datacube
from datacube.storage import BandInfo
from datacube.testutils import gen_tiff_dataset, mk_sample_dataset, mk_test_image
from zarr_io.driver import (
    ZarrDataSource,
    ZarrReaderDriver,
    ZarrWriterDriver,
    file_writer_driver_init,
    reader_driver_init,
    s3_writer_driver_init,
    uri_split,
)
from zarr_io.zarr_io import ZarrIO

S3_ROOT = "s3://mock-bucket/mock-dir/mock-subdir"
'''Mock s3 root object.'''


@mock.patch.dict(environ, {
    'AWS_ACCESS_KEY_ID': 'mock-key-id',
    'AWS_SECRET_ACCESS_KEY': 'mock-secret'
})
@pytest.fixture(scope='module')
def s3():
    '''Mock s3 client.'''
    with mock_s3():
        client = boto3.client('s3', region_name='mock-region')
        client.create_bucket(Bucket='mock-bucket')
        yield client


@pytest.fixture(scope="module")
def data():
    '''Random test data.'''
    return DataArray(np.random.randn(1300, 1300))


@pytest.fixture
def dataset(tmpdir):
    '''Datacube Dataset with random data.

    Based on datacube-core/tests/test_load_data.py'''
    tmpdir = Path(str(tmpdir))

    spatial = dict(resolution=(15, -15),
                   offset=(11230, 1381110),)

    nodata = -999
    array = mk_test_image(96, 64, 'int16', nodata=nodata)

    ds, gbox = gen_tiff_dataset([SimpleNamespace(name='aa', values=array,
                                                 nodata=nodata)],
                                tmpdir,
                                prefix='ds1-',
                                timestamp='2018-07-19',
                                **spatial)
    sources = Datacube.group_datasets([ds], 'time')
    mm = ['aa']
    mm = [ds.type.measurements[k] for k in mm]
    dc_dataset = Datacube.load_data(sources, gbox, mm)

    # Flattening atributes: Zarr doesn't allow dicts
    for var_name in dc_dataset.data_vars:
        data_var = dc_dataset.data_vars[var_name]
        if 'spectral_definition' in data_var.attrs:
            spectral_definition = data_var.attrs.pop('spectral_definition', None)
            data_var.attrs['dc_spectral_definition_response'] = spectral_definition['response']
            data_var.attrs['dc_spectral_definition_wavelength'] = spectral_definition['wavelength']

    # Renaming units: units is a reserved name in Xarray coordinates
    for var_name in dc_dataset.coords:
        coord_var = dc_dataset.coords[var_name]
        if 'units' in coord_var.attrs:
            units = coord_var.attrs.pop('units', None)
            coord_var.attrs['dc_units'] = units

    return dc_dataset


@pytest.fixture
def odc_dataset(dataset, tmpdir):
    '''Write xr.dataset to zarr files.'''
    root = Path(tmpdir) / 'data'
    group_name = list(dataset.keys())[0]
    zio = ZarrIO(protocol='file')
    zio.save_dataset(root=root,
                     group_name=group_name,
                     relative=False,
                     dataset=dataset,
                     chunks={'x': 50, 'y': 50})
    bands = [{
        'name': group_name,
        'path': str(root / group_name)
    }]
    ds1 = mk_sample_dataset(bands, 'file', format='zarr')
    return ds1


def test_datasource(dataset, odc_dataset):
    '''Test ZarrDataSource.

    Data is saved to file and opened by the data source.'''
    group_name = list(dataset.keys())[0]
    source = ZarrDataSource(BandInfo(odc_dataset, group_name))
    with source.open() as band_source:
        ds = band_source.read()
        assert np.array_equal(ds, dataset[group_name].values)


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
    assert np.array_equal(ds2, dataset[group_name].values)

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


def _check_zarr_files(root, data):
    '''Check zarr files in local filesystem.

    Only some metadata and chunk file names are checked, not actual binary content.'''
    assert root.exists(), f'Missing {root} after save'
    # Check chunks in root level metadata
    metadata_path = root / '.zmetadata'
    assert metadata_path.exists(), f'Missing .zmetadata in {root}'
    with metadata_path.open() as fh:
        metadata = load(fh)
    assert metadata['metadata']['dataset1/array1/.zarray']['chunks'] == [
        1000,
        1100
    ], 'Chunks not as set'

    dataset_dir = root / 'dataset1'
    assert dataset_dir.exists(), f'Missing dataset1/ in {root}'
    array_dir = dataset_dir / 'array1'
    assert array_dir.exists(), f'Missing array1/ in {dataset_dir}'

    # Check chunks in array level metadata
    metadata_path = array_dir / '.zarray'
    assert metadata_path.exists(), f'Missing .zarray in {array_dir}'
    with metadata_path.open() as fh:
        metadata = load(fh)
    assert metadata['chunks'] == [1000, 1100], 'Chunks not as set'
    assert metadata['shape'] == list(data.shape), 'Data shape not as set'

    # Check chunk files
    chunk_files = sorted([path.name for path in array_dir.glob('?.?')])
    assert chunk_files == ['0.0', '0.1', '1.0', '1.1'], 'Unexpected chunk files'


def _save(storage, data, root):
    '''Save data to storage.'''
    zio = ZarrIO(protocol=storage)
    # Clean storage area
    zio.clean_store(root=root)  # TODO: move to its own test
    # Persist to file
    zio.save_dataset(root=root,
                     group_name='dataset1',
                     relative=True,
                     dataset=data.to_dataset(name='array1'),
                     chunks={'dim_0': 1000, 'dim_1': 1100})


@pytest.mark.parametrize('storage', ('file', 's3'))
def test_save(storage, data, tmpdir, s3):  # s3 param not used but required for mock s3
    '''Test zarr save and load.'''
    root = S3_ROOT if storage == 's3' else Path(tmpdir) / 'data'
    _save(storage, data, root)

    if storage == 'file':
        _check_zarr_files(root, data)

    # Load data and check it hasn't changed
    zio = ZarrIO(protocol=storage)
    ds = zio.load_dataset(root=root, group_name='dataset1', relative=True)
    assert np.array_equal(data, ds.array1.values)


@pytest.mark.parametrize('storage', ('file', 's3'))
def test_print_tree(storage, data, tmpdir, s3):  # s3 param not used but required for mock s3
    '''Test zarr print data tree.'''
    root = S3_ROOT if storage == 's3' else Path(tmpdir) / 'data'
    _save(storage, data, root)

    zio = ZarrIO(protocol=storage)
    actual = str(zio.print_tree(root))
    expected = '''/
 └── dataset1
     └── array1 (1300, 1300) float64'''
    assert actual == expected


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
        assert np.array_equal(ds, dataset[group_name].values)

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
