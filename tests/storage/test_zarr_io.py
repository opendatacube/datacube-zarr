from json import load
from os import environ
from pathlib import Path

import pytest
import boto3
import mock
import numpy as np
from moto import mock_s3
from xarray import DataArray

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
        s3 = boto3.client('s3', region_name='mock-region')
        s3.create_bucket(Bucket='mock-bucket')
        yield s3


@pytest.fixture(scope="module")
def data():
    '''Random test data.'''
    return DataArray(np.random.randn(1300, 1300))


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
    assert metadata['chunks'] == [
                1000,
                1100
            ], 'Chunks not as set'
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


@pytest.mark.parametrize('storage', ('file', ))
def test_print_tree(storage, data, tmpdir, s3):  # s3 param not used but required for mock s3
    '''Test zarr print data tree.'''
    root = S3_ROOT if storage == 's3' else Path(tmpdir) / 'data'
    _save(storage, data, root)

    zio = ZarrIO(protocol='file')
    s1 = str(zio.print_tree(root))
    s2 = '''/
 └── dataset1
     └── array1 (1300, 1300) float64'''
    assert s1 == s2
