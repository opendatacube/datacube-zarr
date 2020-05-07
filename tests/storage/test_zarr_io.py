'''Low-level zarr_io tests.'''
from json import load
from os import environ
from pathlib import Path
from random import random, sample
from types import SimpleNamespace

import pytest
import boto3
import mock
import numpy as np
from mock import MagicMock
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
from zarr_io.zarr_io import ZarrBase, ZarrIO

CHUNKS = (
    {  # When no chunk set, xarray and zarr decide. For a 1300x1300 data, it is:
        'input': None,
        'chunks_per_side': 4,
        'output': [325, 325]
    },
    {  # User specified chunks, input and output should match
        'input': {'dim_0': 1000, 'dim_1': 1100},
        'chunks_per_side': 2,
        'output': [1000, 1100]
    }
)
'''Zarr chunk sizes to be tested and expected output in metadata and number of
chunks per side.'''

SPECTRAL_DEFINITION = {
    'wavelength': sorted(sample(range(380, 750), 150)),
    'response': [random() for i in range(150)]
}
'''Random spectral definition with 150 values.'''

count = 0
'''Give a new ID to each moto bucket as they don't seem to clean properly between
runs.'''


@mock.patch.dict(environ, {
    'AWS_ACCESS_KEY_ID': 'mock-key-id',
    'AWS_SECRET_ACCESS_KEY': 'mock-secret'
})
@pytest.fixture
def s3():
    '''Mock s3 client and root url.'''
    global count
    with mock_s3():
        client = boto3.client('s3', region_name='mock-region')
        bucket_name = f'mock-bucket-{count}'
        count += 1
        client.create_bucket(Bucket=bucket_name)
        root = f's3://{bucket_name}/mock-dir/mock-subdir'
        yield {'client': client, 'root': root}


@pytest.fixture
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


def _check_zarr_files(data, root, group_name, name, relative, chunks):
    '''Check zarr files in local filesystem.

    Only some metadata and chunk file names are checked, not actual binary content.'''
    if not relative:
        root = root / group_name
    assert root.exists(), f'Missing {root} after save'
    # Check chunks in root level metadata
    metadata_path = root / '.zmetadata'
    assert metadata_path.exists(), f'Missing .zmetadata in {root}'
    with metadata_path.open() as fh:
        metadata = load(fh)
    assert metadata['metadata'][f'{group_name}/{name}/.zarray']['chunks'] == \
        chunks['output'], 'Chunks not as set'

    dataset_dir = root / group_name
    assert dataset_dir.exists(), f'Missing {group_name}/ in {root}'
    array_dir = dataset_dir / name
    assert array_dir.exists(), f'Missing {name}/ in {dataset_dir}'

    # Check chunks in array level metadata
    metadata_path = array_dir / '.zarray'
    assert metadata_path.exists(), f'Missing .zarray in {array_dir}'
    with metadata_path.open() as fh:
        metadata = load(fh)
    assert metadata['chunks'] == chunks['output'], 'Chunks not as set'
    assert metadata['shape'] == list(data.shape), 'Data shape not as set'

    # Check chunk files
    chunk_files = sorted([path.name for path in array_dir.glob('?.?')])
    expected_chunk_files = sorted([f'{i}.{j}'
                                   for i in range(chunks['chunks_per_side'])
                                   for j in range(chunks['chunks_per_side'])])
    assert chunk_files == expected_chunk_files, 'Unexpected chunk files'


def _save_dataarray(data, protocol, root, group_name, name, relative, chunks):
    '''Save DataArray to storage.'''
    zio = ZarrIO(protocol=protocol)
    zio.save_dataarray(root=str(root),
                       group_name=group_name,
                       dataarray=data.copy(),
                       name=name,
                       chunks=chunks,
                       relative=relative)


def _save_dataset(data, protocol, root, group_name, name, relative, chunks):
    '''Save Dataset to storage.'''
    zio = ZarrIO(protocol=protocol)
    zio.save_dataset(root=str(root),
                     group_name=group_name,
                     relative=relative,
                     dataset=data.copy().to_dataset(name=name),
                     chunks=chunks)


def _load_dataset(protocol, root, group_name, relative):
    '''Save Dataset to storage.'''
    zio = ZarrIO(protocol=protocol)
    return zio.load_dataset(root=str(root),
                            group_name=group_name,
                            relative=relative)


@pytest.mark.parametrize('protocol', ('file', 's3'))
@pytest.mark.parametrize('relative', (True, False))
@pytest.mark.parametrize('chunks', CHUNKS)
def test_save_dataarray(protocol, relative, chunks, data, tmpdir, s3):
    '''Test ZarrIO.save_dataarray save and load for a single DataArray.'''
    root = s3['root'] if protocol == 's3' else Path(tmpdir) / 'data'
    group_name = 'dataset1'
    name = 'array1'
    _save_dataarray(data, protocol, root, group_name, name, relative, chunks['input'])

    if protocol == 'file':
        # Check filesystem structure and some metadata
        _check_zarr_files(data, root, group_name, name, relative, chunks)

    # Load and check data
    ds = _load_dataset(protocol, root, group_name, relative)
    assert np.array_equal(data, ds[name].values)


@pytest.mark.parametrize('protocol', ('file', 's3'))
@pytest.mark.parametrize('relative', (True, False))
@pytest.mark.parametrize('chunks', CHUNKS)
def test_save_dataset(protocol, relative, chunks, data, tmpdir, s3):
    '''Test ZarrIO.save_dataset save and load for a single Dataset.'''
    root = s3['root'] if protocol == 's3' else Path(tmpdir) / 'data'
    group_name = 'dataset1'
    name = 'array1'
    _save_dataset(data, protocol, root, group_name, name, relative, chunks['input'])

    if protocol == 'file':
        # Check filesystem structure and some metadata
        _check_zarr_files(data, root, group_name, name, relative, chunks)

    # Load and check data
    ds = _load_dataset(protocol, root, group_name, relative)
    assert np.array_equal(data, ds[name].values)


@pytest.mark.parametrize('protocol', ('file', 's3'))
@pytest.mark.parametrize('relative', (True, False))
@pytest.mark.parametrize('chunks', CHUNKS)
def test_save_dataarrays(protocol, relative, chunks, data, tmpdir, s3):
    '''Test ZarrIO.save_dataarray save and load for a single DataArray.'''
    root = s3['root'] if protocol == 's3' else Path(tmpdir) / 'data'
    datasets = {}
    for set_no in range(1, 3):
        group_name = f'dataset{set_no}'
        name = f'array{set_no}'
        ds = data.copy() * set_no  # Make each dataset a bit different for testing
        datasets[set_no] = ds
        _save_dataarray(ds, protocol, root, group_name, name, relative, chunks['input'])

    # Load and check data
    for set_no, dataset in datasets.items():
        group_name = f'dataset{set_no}'
        name = f'array{set_no}'
        ds = _load_dataset(protocol, root, group_name, relative)
        assert np.array_equal(dataset, ds[name].values)


@pytest.mark.parametrize('protocol', ('file', 's3'))
@pytest.mark.parametrize('relative', (True, False))
@pytest.mark.parametrize('chunks', CHUNKS)
def test_save_datasets(protocol, relative, chunks, data, tmpdir, s3):
    '''Test ZarrIO.save_dataset save and load for a single Dataset.'''
    root = s3['root'] if protocol == 's3' else Path(tmpdir) / 'data'
    datasets = {}
    for set_no in range(1, 3):
        group_name = f'dataset{set_no}'
        name = f'array{set_no}'
        ds = data.copy() * set_no  # Make each dataset a bit different for testing
        datasets[set_no] = ds
        _save_dataset(ds, protocol, root, group_name, name, relative, chunks['input'])

    # Load and check data
    for set_no, dataset in datasets.items():
        group_name = f'dataset{set_no}'
        name = f'array{set_no}'
        ds = _load_dataset(protocol, root, group_name, relative)
        assert np.array_equal(dataset, ds[name].values)


@pytest.mark.parametrize('protocol', ('file', 's3'))
@pytest.mark.parametrize('relative', (True, False))
def test_print_tree(protocol, relative, data, tmpdir, s3):  # s3 param not used but required for mock s3
    '''Test zarr print data tree.'''
    root = s3['root'] if protocol == 's3' else Path(tmpdir) / 'data'
    zio = ZarrIO(protocol=protocol)
    zio.save_dataarray(root=root, group_name='dataset1', dataarray=data,
                       name='array1', chunks=CHUNKS[1]['input'], relative=relative)
    zio.save_dataarray(root=root, group_name='dataset2', dataarray=data,
                       name='array2', chunks=CHUNKS[1]['input'], relative=relative)
    zio.save_dataset(root=root, group_name='dataset3',
                     dataset=data.to_dataset(name='array1'),
                     chunks=CHUNKS[1]['input'], relative=relative)
    zio.save_dataset(root=root, group_name='dataset4',
                     dataset=data.to_dataset(name='array2'),
                     chunks=CHUNKS[1]['input'], relative=relative)
    actual = str(zio.print_tree(root))
    if relative:
        expected = '''/
 ├── dataset1
 │   └── array1 (1300, 1300) float64
 ├── dataset2
 │   └── array2 (1300, 1300) float64
 ├── dataset3
 │   └── array1 (1300, 1300) float64
 └── dataset4
     └── array2 (1300, 1300) float64'''
    else:
        expected = '''/
 ├── dataset1
 │   └── dataset1
 │       └── array1 (1300, 1300) float64
 ├── dataset2
 │   └── dataset2
 │       └── array2 (1300, 1300) float64
 ├── dataset3
 │   └── dataset3
 │       └── array1 (1300, 1300) float64
 └── dataset4
     └── dataset4
         └── array2 (1300, 1300) float64'''
    assert actual == expected


@pytest.mark.parametrize('protocol', ('file', 's3'))
def test_clean_store(protocol, data, tmpdir, s3):
    '''Test cleaning of zarr store.'''
    relative = True
    root = s3['root'] if protocol == 's3' else Path(tmpdir) / 'data'
    zio = ZarrIO(protocol=protocol)
    zio.save_dataset(root=root, group_name='dataset1',
                     dataset=data.to_dataset(name='array1'),
                     chunks=CHUNKS[1]['input'], relative=relative)
    assert str(zio.print_tree(root)) == '''/
 └── dataset1
     └── array1 (1300, 1300) float64'''
    # Clean and store something else
    zio.clean_store(root)
    zio.save_dataarray(root=root, group_name='dataset2', dataarray=data,
                       name='array2', chunks=CHUNKS[1]['input'], relative=relative)
    assert str(zio.print_tree(root)) == '''/
 └── dataset2
     └── array2 (1300, 1300) float64'''


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


def test_zarr_other_writer_driver():
    '''Testing a loophole where the protocol gets updated late.'''
    writer = file_writer_driver_init()
    writer.zio.protocol = 'xxx'
    assert writer.aliases == []


def test_invalid_protocol():
    with pytest.raises(ValueError) as excinfo:
        ZarrBase(protocol='xxx')
    assert str(excinfo.value) == 'unknown protocol: xxx'

    with pytest.raises(ValueError) as excinfo:
        ZarrIO(protocol='xxx')
    assert str(excinfo.value) == 'unknown protocol: xxx'

    with pytest.raises(ValueError) as excinfo:
        ZarrWriterDriver(protocol='xxx')
    assert str(excinfo.value) == 'unknown protocol: xxx'


# @pytest.mark.skip(reason='Negative interaction with other tests, presumably through s3')
@pytest.mark.parametrize('protocol', ('file', 's3'))
def test_zarr_file_writer_driver_save(protocol, data, tmpdir, s3):
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
        storage_config={'chunking': CHUNKS[1]['input']}
    )
    if protocol == 'file':
        _check_zarr_files(data, root, group_name, name, relative, CHUNKS[1])
    # Load and check data
    ds_out = _load_dataset(protocol, root, group_name, relative=relative)
    assert ds_in.equals(ds_out)  # Compare values only


def test_zarr_file_writer_driver_data_corrections(data, tmpdir):
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
        storage_config={'chunking': CHUNKS[1]['input']}
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
