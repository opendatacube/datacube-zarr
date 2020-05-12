'''Unit tests for the zarr_io.zarr_io module.'''
from pathlib import Path

import pytest
import numpy as np

from zarr_io.zarr_io import ZarrBase, ZarrIO

from ..utils import _check_zarr_files, _load_dataset, _save_dataarray, _save_dataset


@pytest.mark.parametrize('protocol', ('file', 's3'))
@pytest.mark.parametrize('relative', (True, False))
def test_save_dataarray(protocol, relative, chunks, data, tmpdir, s3):
    '''Test ZarrIO.save_dataarray to save and load for a single DataArray.'''
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
def test_save_dataset(protocol, relative, chunks, data, tmpdir, s3):
    '''Test ZarrIO.save_dataset to save and load for a single Dataset.'''
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
def test_save_dataarrays(protocol, relative, chunks, data, tmpdir, s3):
    '''Test ZarrIO.save_dataarray to save and load multiple DataArrays.'''
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
def test_save_datasets(protocol, relative, chunks, data, tmpdir, s3):
    '''Test ZarrIO.save_dataset save and load multiple Datasets.'''
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
def test_print_tree(protocol, relative, fixed_chunks, data, tmpdir, s3):
    '''Test zarr print data tree with a mix of Datasets and DataArrays co-existing.'''
    root = s3['root'] if protocol == 's3' else Path(tmpdir) / 'data'
    zio = ZarrIO(protocol=protocol)
    zio.save_dataarray(root=root, group_name='dataset1', dataarray=data,
                       name='array1', chunks=fixed_chunks['input'], relative=relative)
    zio.save_dataarray(root=root, group_name='dataset2', dataarray=data,
                       name='array2', chunks=fixed_chunks['input'], relative=relative)
    zio.save_dataset(root=root, group_name='dataset3',
                     dataset=data.to_dataset(name='array1'),
                     chunks=fixed_chunks['input'], relative=relative)
    zio.save_dataset(root=root, group_name='dataset4',
                     dataset=data.to_dataset(name='array2'),
                     chunks=fixed_chunks['input'], relative=relative)
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
def test_clean_store(protocol, fixed_chunks, data, tmpdir, s3):
    '''Test cleaning of zarr store.'''
    relative = True
    root = s3['root'] if protocol == 's3' else Path(tmpdir) / 'data'
    zio = ZarrIO(protocol=protocol)
    zio.save_dataset(root=root, group_name='dataset1',
                     dataset=data.to_dataset(name='array1'),
                     chunks=fixed_chunks['input'], relative=relative)
    assert str(zio.print_tree(root)) == '''/
 └── dataset1
     └── array1 (1300, 1300) float64'''
    # Clean and store something else
    zio.clean_store(root)
    zio.save_dataarray(root=root, group_name='dataset2', dataarray=data,
                       name='array2', chunks=fixed_chunks['input'], relative=relative)
    assert str(zio.print_tree(root)) == '''/
 └── dataset2
     └── array2 (1300, 1300) float64'''


def test_invalid_protocol():
    '''Test exceptions when an invalid protocol is used.'''
    with pytest.raises(ValueError) as excinfo:
        ZarrBase(protocol='xxx')
    assert str(excinfo.value) == 'unknown protocol: xxx'

    with pytest.raises(ValueError) as excinfo:
        ZarrIO(protocol='xxx')
    assert str(excinfo.value) == 'unknown protocol: xxx'


@pytest.mark.parametrize('protocol', ('file', 's3'))
@pytest.mark.parametrize('relative', (True, False))
def test_invalid_mode(protocol, relative, fixed_chunks, data, tmpdir, s3):
    '''Test exceptions when an invalid mode is used.'''
    root = s3['root'] if protocol == 's3' else Path(tmpdir) / 'data'
    zio = ZarrIO(protocol=protocol)

    with pytest.raises(ValueError) as excinfo:
        zio.save_dataset(root=root,
                         group_name='dataset1',
                         dataset=data.to_dataset(name='array1'),
                         chunks=fixed_chunks['input'],
                         mode='xxx',
                         relative=relative)
    assert str(excinfo.value) == f"Only the following modes are supported {ZarrIO.WRITE_MODES}"

    with pytest.raises(ValueError) as excinfo:
        zio.save_dataarray(root=root,
                           group_name='dataset2',
                           dataarray=data,
                           name='array1',
                           chunks=fixed_chunks['input'],
                           mode='xxx',
                           relative=relative)
    assert str(excinfo.value) == f"Only the following modes are supported {ZarrIO.WRITE_MODES}"


@pytest.mark.parametrize('protocol', ('file', 's3'))
@pytest.mark.parametrize('relative', (True, False))
def test_overwrite_dataset(protocol, relative, fixed_chunks, data, tmpdir, s3):
    '''Test overwriting an existing dataset.'''
    root = s3['root'] if protocol == 's3' else Path(tmpdir) / 'data'
    group_name = 'dataset_group_name'
    zio = ZarrIO(protocol=protocol)

    # write the dataset twice
    for i in range(2):
        name = f'array{i}'
        dataset = (data.copy() + i).to_dataset(name=name)
        zio.save_dataset(
            root=root,
            group_name=group_name,
            dataset=dataset,
            chunks=fixed_chunks['input'],
            mode='w',
            relative=relative
        )

    ds = zio.load_dataset(
        root=str(root), group_name=group_name, relative=relative
    )
    np.array_equal(dataset[name], ds[name].values)
