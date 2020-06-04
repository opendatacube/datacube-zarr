'''Unit tests for the zarr_io.zarr_io module.'''
from pathlib import Path

import pytest
import numpy as np
import xarray as xr

from zarr_io.zarr_io import ZarrBase, ZarrIO

from ..utils import _check_zarr_files, _load_dataset, _save_dataarray, _save_dataset


@pytest.mark.parametrize('protocol', ('file', 's3'))
def test_save_dataarray(protocol, chunks, data, tmpdir, s3):
    '''Test ZarrIO.save_dataarray to save and load for a single DataArray.'''
    root = s3['root'] if protocol == 's3' else Path(tmpdir) / 'data.zarr'
    group_name = 'dataset1'
    name = 'array1'
    uri = f'{protocol}://{root}#{group_name}'
    _save_dataarray(data, uri, name, chunks['input'])

    if protocol == 'file':
        # Check filesystem structure and some metadata
        _check_zarr_files(data, root, group_name, name, chunks)

    # Load and check data
    ds = _load_dataset(uri)
    assert np.array_equal(data, ds[name].values)


@pytest.mark.parametrize('protocol', ('file', 's3'))
def test_save_dataset(protocol, chunks, data, tmpdir, s3):
    '''Test ZarrIO.save_dataset to save and load for a single Dataset.'''
    root = s3['root'] if protocol == 's3' else Path(tmpdir) / 'data.zarr'
    group_name = 'dataset1'
    name = 'array1'
    uri = f'{protocol}://{root}#{group_name}'
    _save_dataset(data, uri, name, chunks['input'])

    if protocol == 'file':
        # Check filesystem structure and some metadata
        _check_zarr_files(data, root, group_name, name, chunks)

    # Load and check data
    ds = _load_dataset(uri)
    assert np.array_equal(data, ds[name].values)


@pytest.mark.parametrize('protocol', ('file', 's3'))
def test_save_dataarrays(protocol, chunks, data, tmpdir, s3):
    '''Test ZarrIO.save_dataarray to save and load multiple DataArrays.'''
    root = s3['root'] if protocol == 's3' else Path(tmpdir) / 'data.zarr'
    datasets = {}
    for set_no in range(1, 3):
        group_name = f'dataset{set_no}'
        uri = f'{protocol}://{root}#{group_name}'
        name = f'array{set_no}'
        ds = data.copy() * set_no  # Make each dataset a bit different for testing
        datasets[set_no] = ds
        _save_dataarray(ds, uri, name, chunks['input'])

    # Load and check data
    for set_no, dataset in datasets.items():
        group_name = f'dataset{set_no}'
        uri = f'{protocol}://{root}#{group_name}'
        name = f'array{set_no}'
        ds = _load_dataset(uri)
        assert np.array_equal(dataset, ds[name].values)


@pytest.mark.parametrize('protocol', ('file', 's3'))
def test_save_datasets(protocol, chunks, data, tmpdir, s3):
    '''Test ZarrIO.save_dataset save and load multiple Datasets.'''
    root = s3['root'] if protocol == 's3' else Path(tmpdir) / 'data.zarr'
    datasets = {}
    for set_no in range(1, 3):
        group_name = f'dataset{set_no}'
        uri = f'{protocol}://{root}#{group_name}'
        name = f'array{set_no}'
        ds = data.copy() * set_no  # Make each dataset a bit different for testing
        datasets[set_no] = ds
        _save_dataset(ds, uri, name, chunks['input'])

    # Load and check data
    for set_no, dataset in datasets.items():
        group_name = f'dataset{set_no}'
        uri = f'{protocol}://{root}#{group_name}'
        name = f'array{set_no}'
        ds = _load_dataset(uri)
        assert np.array_equal(dataset, ds[name].values)


@pytest.mark.parametrize('protocol', ('file', 's3'))
def test_print_tree(protocol, fixed_chunks, data, tmpdir, s3):
    '''Test zarr print data tree with a mix of Datasets and DataArrays co-existing.'''
    root = s3['root'] if protocol == 's3' else Path(tmpdir) / 'data.zarr'
    root_uri = f'{protocol}://{root}'
    zio = ZarrIO()
    zio.save_dataarray(uri=f'{root_uri}#dataset1', dataarray=data,
                       name='array1', chunks=fixed_chunks['input'])
    zio.save_dataarray(uri=f'{root_uri}#dataset2', dataarray=data,
                       name='array2', chunks=fixed_chunks['input'])
    zio.save_dataset(uri=f'{root_uri}#dataset3',
                     dataset=data.to_dataset(name='array1'),
                     chunks=fixed_chunks['input'])
    zio.save_dataset(uri=f'{root_uri}#dataset4',
                     dataset=data.to_dataset(name='array2'),
                     chunks=fixed_chunks['input'])
    actual = str(zio.print_tree(root_uri))
    expected = '''/
 ├── dataset1
 │   └── array1 (1300, 1300) float64
 ├── dataset2
 │   └── array2 (1300, 1300) float64
 ├── dataset3
 │   └── array1 (1300, 1300) float64
 └── dataset4
     └── array2 (1300, 1300) float64'''
    assert actual == expected


@pytest.mark.parametrize('protocol', ('file', 's3'))
def test_clean_store(protocol, fixed_chunks, data, tmpdir, s3):
    '''Test cleaning of zarr store.'''
    root = s3['root'] if protocol == 's3' else Path(tmpdir) / 'data.zarr'
    root_uri = f'{protocol}://{root}'
    zio = ZarrIO()
    zio.save_dataset(uri=f'{root_uri}#dataset1',
                     dataset=data.to_dataset(name='array1'),
                     chunks=fixed_chunks['input'])
    assert str(zio.print_tree(root_uri)) == '''/
 └── dataset1
     └── array1 (1300, 1300) float64'''
    # Clean and store something else
    zio.clean_store(root_uri)
    zio.save_dataarray(uri=f'{root_uri}#dataset2', dataarray=data,
                       name='array2', chunks=fixed_chunks['input'])
    assert str(zio.print_tree(root_uri)) == '''/
 └── dataset2
     └── array2 (1300, 1300) float64'''


def test_invalid_protocol():
    '''Test exceptions when an invalid protocol is used.'''
    with pytest.raises(ValueError) as excinfo:
        zio = ZarrIO()
        zio.get_root(f'xxx://root')
    assert str(excinfo.value) == 'unknown protocol: xxx'


@pytest.mark.parametrize('protocol', ('file', 's3'))
def test_invalid_mode(protocol, fixed_chunks, data, tmpdir, s3):
    '''Test exceptions when an invalid mode is used.'''
    root = s3['root'] if protocol == 's3' else Path(tmpdir) / 'data.zarr'
    zio = ZarrIO()

    with pytest.raises(ValueError) as excinfo:
        zio.save_dataset(f'{protocol}://{root}#dataset1',
                         dataset=data.to_dataset(name='array1'),
                         chunks=fixed_chunks['input'],
                         mode='xxx')
    assert str(excinfo.value) == f"Only the following modes are supported {ZarrIO.WRITE_MODES}"

    with pytest.raises(ValueError) as excinfo:
        zio.save_dataarray(f'{protocol}://{root}#dataset2',
                           dataarray=data,
                           name='array1',
                           chunks=fixed_chunks['input'],
                           mode='xxx')
    assert str(excinfo.value) == f"Only the following modes are supported {ZarrIO.WRITE_MODES}"


@pytest.mark.parametrize('protocol', ('file', 's3'))
def test_overwrite_dataset(protocol, fixed_chunks, data, tmpdir, s3):
    '''Test overwriting an existing dataset.'''
    root = s3['root'] if protocol == 's3' else Path(tmpdir) / 'data.zarr'
    group_name = 'dataset_group_name'
    uri = f'{protocol}://{root}#{group_name}'
    zio = ZarrIO()

    # write the dataset twice
    for i in range(2):
        name = f'array{i}'
        dataset = (data.copy() + i).to_dataset(name=name)
        zio.save_dataset(
            uri=uri,
            dataset=dataset,
            chunks=fixed_chunks['input'],
            mode='w',
        )

    ds = zio.load_dataset(uri=uri)
    np.array_equal(dataset[name], ds[name].values)


@pytest.mark.parametrize('protocol', ('file', 's3'))
def test_save_datasets_nested(protocol, data, tmpdir, s3):
    '''Test saving nested datasets (i.e. datasets and groups side by side).'''
    root = s3['root'] if protocol == 's3' else Path(tmpdir) / 'data.zarr'
    groups = ["", "group1A", "group1B", "group1A/group2A", "group1A/group2B"]
    datasets = {}
    for i in range(len(groups)):
        name = f'array{i}'
        ds = data.copy() * (i + 1)  # Make each dataset a bit different for testing
        datasets[name] = ds
        uri = f'{protocol}://{root}#{groups[i]}'
        _save_dataset(ds, uri, name)

    # Load and check data
    for i, (name, dataset) in enumerate(datasets.items()):
        uri = f'{protocol}://{root}#{groups[i]}'
        ds = _load_dataset(uri)
        assert np.array_equal(dataset, ds[name].values)


@pytest.mark.parametrize('protocol', ('file', 's3'))
def test_save_datasets_nested_zarr(protocol, data, tmpdir, s3):
    '''Test saving nested zarr files.'''
    root = s3['root'] if protocol == 's3' else Path(tmpdir) / 'data.zarr'
    datasets = {}
    roots = [str(root), f"{str(root)}/group.zarr"]

    for i, r in enumerate(roots):
        name = f'array{i}'
        ds = data.copy() * (i + 1)  # Make each dataset a bit different for testing
        datasets[name] = ds
        uri = f'{protocol}://{r}'
        _save_dataset(ds, uri, name)

    # Load and check data
    for i, (name, dataset) in enumerate(datasets.items()):

        # load each zarr directly
        r = roots[i]
        uri = f'{protocol}://{r}'
        ds = _load_dataset(uri)
        assert np.array_equal(dataset, ds[name].values)

        # Load nested zarr as group of root zarr, with consolidated = False
        store = ZarrIO().get_root(uri)
        group = r.split(str(root))[0]
        if group:
            ds = xr.open_zarr(store=store, group=group, consolidated=False)
            assert np.array_equal(dataset, ds[name].values)
