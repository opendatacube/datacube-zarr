'''Low-level zarr_io tests.'''
from json import load
from os import environ
from pathlib import Path
from types import SimpleNamespace

import pytest
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


def test_datasource(dataset, tmpdir):
    '''Test the ZarrDataSource.'''
    # Write xr.dataset to file
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
    # Create data source and compare read data
    source = ZarrDataSource(BandInfo(ds1, group_name))
    with source.open() as rdr:
        ds2 = rdr.read()
    assert np.array_equal(ds2, dataset.aa.values)


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


def test_zarr_reader_driver():
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
