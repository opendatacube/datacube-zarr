'''Unit tests for the datacube_zarr.zarr_io module.'''

import pytest
import numpy as np
import xarray as xr

from datacube_zarr.zarr_io import ZarrIO, replace_dataset_dim

from .utils import _check_zarr_files, _load_dataset, _save_dataarray, _save_dataset


def test_save_dataarray(uri, s3, chunks, data):
    '''Test ZarrIO.save_dataarray to save and load for a single DataArray.'''
    name = 'array1'
    _save_dataarray(data, uri, name, chunks['input'])

    # Check filesystem or s3 structure and some metadata
    _check_zarr_files(data, uri, name, chunks, s3)

    # Load and check data
    ds = _load_dataset(uri)
    assert np.array_equal(data, ds[name].values)


def test_save_dataset(uri, chunks, data, s3):
    '''Test ZarrIO.save_dataset to save and load for a single Dataset.'''
    name = 'array1'
    _save_dataset(data, uri, name, chunks['input'])

    # Check filesystem or s3 structure and some metadata
    _check_zarr_files(data, uri, name, chunks, s3)

    # Load and check data
    ds = _load_dataset(uri)
    assert np.array_equal(data, ds[name].values)


def test_save_dataarrays(uri, chunks, data):
    '''Test ZarrIO.save_dataarray to save and load multiple DataArrays.'''
    datasets = {}
    for set_no in range(1, 3):
        _uri = f'{uri[:-1]}{set_no}'
        name = f'array{set_no}'
        ds = data.copy() * set_no  # Make each dataset a bit different for testing
        datasets[set_no] = ds
        _save_dataarray(ds, _uri, name, chunks['input'])

    # Load and check data
    for set_no, dataset in datasets.items():
        _uri = f'{uri[:-1]}{set_no}'
        name = f'array{set_no}'
        ds = _load_dataset(_uri)
        assert np.array_equal(dataset, ds[name].values)


def test_save_datasets(uri, chunks, data):
    '''Test ZarrIO.save_dataset save and load multiple Datasets.'''
    datasets = {}
    for set_no in range(1, 3):
        _uri = f'{uri[:-1]}{set_no}'
        name = f'array{set_no}'
        ds = data.copy() * set_no  # Make each dataset a bit different for testing
        datasets[set_no] = ds
        _save_dataset(ds, _uri, name, chunks['input'])

    # Load and check data
    for set_no, dataset in datasets.items():
        _uri = f'{uri[:-1]}{set_no}'
        name = f'array{set_no}'
        ds = _load_dataset(_uri)
        assert np.array_equal(dataset, ds[name].values)


def test_print_tree(uri, fixed_chunks, data):
    '''Test zarr print data tree with a mix of Datasets and DataArrays co-existing.'''
    root_uri = uri.rsplit('#', 1)[0]
    zio = ZarrIO()
    zio.save_dataarray(
        uri=f'{root_uri}#dataset1',
        dataarray=data,
        name='array1',
        chunks=fixed_chunks['input'],
    )
    zio.save_dataarray(
        uri=f'{root_uri}#dataset2',
        dataarray=data,
        name='array2',
        chunks=fixed_chunks['input'],
    )
    zio.save_dataset(
        uri=f'{root_uri}#dataset3',
        dataset=data.to_dataset(name='array1'),
        chunks=fixed_chunks['input'],
    )
    zio.save_dataset(
        uri=f'{root_uri}#dataset4',
        dataset=data.to_dataset(name='array2'),
        chunks=fixed_chunks['input'],
    )
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


def test_clean_store(uri, fixed_chunks, data):
    '''Test cleaning of zarr store.'''
    root_uri = uri.rsplit('#', 1)[0]
    zio = ZarrIO()
    zio.save_dataset(
        uri=f'{root_uri}#dataset1',
        dataset=data.to_dataset(name='array1'),
        chunks=fixed_chunks['input'],
    )
    assert (
        str(zio.print_tree(root_uri))
        == '''/
 └── dataset1
     └── array1 (1300, 1300) float64'''
    )
    # Clean and store something else
    zio.clean_store(root_uri)
    zio.save_dataarray(
        uri=f'{root_uri}#dataset2',
        dataarray=data,
        name='array2',
        chunks=fixed_chunks['input'],
    )
    assert (
        str(zio.print_tree(root_uri))
        == '''/
 └── dataset2
     └── array2 (1300, 1300) float64'''
    )


def test_invalid_protocol():
    '''Test exceptions when an invalid protocol is used.'''
    with pytest.raises(ValueError) as excinfo:
        zio = ZarrIO()
        zio.get_root('xxx://root')
    assert str(excinfo.value) == 'unknown protocol: xxx'


def test_invalid_mode(uri, fixed_chunks, data):
    '''Test exceptions when an invalid mode is used.'''
    zio = ZarrIO()
    with pytest.raises(ValueError) as excinfo:
        zio.save_dataset(
            uri,
            dataset=data.to_dataset(name='array1'),
            chunks=fixed_chunks['input'],
            mode='xxx',
        )
    assert (
        str(excinfo.value)
        == f"Only the following modes are supported {ZarrIO.WRITE_MODES}"
    )

    with pytest.raises(ValueError) as excinfo:
        zio.save_dataarray(
            uri, dataarray=data, name='array1', chunks=fixed_chunks['input'], mode='xxx'
        )
    assert (
        str(excinfo.value)
        == f"Only the following modes are supported {ZarrIO.WRITE_MODES}"
    )


def test_overwrite_dataset(uri, fixed_chunks, data):
    '''Test overwriting an existing dataset.'''
    zio = ZarrIO()
    # write the dataset twice
    for i in range(2):
        name = f'array{i}'
        dataset = (data.copy() + i).to_dataset(name=name)
        zio.save_dataset(uri=uri, dataset=dataset, chunks=fixed_chunks['input'], mode='w')
    ds = zio.load_dataset(uri=uri)
    np.array_equal(dataset[name], ds[name].values)


def test_save_datasets_nested(uri, data):
    '''Test saving nested datasets (i.e. datasets and groups side by side).'''
    root_uri = uri.rsplit('#', 1)[0]
    groups = ["", "group1A", "group1B", "group1A/group2A", "group1A/group2B"]
    datasets = {}
    for i in range(len(groups)):
        name = f'array{i}'
        ds = data.copy() * (i + 1)  # Make each dataset a bit different for testing
        datasets[name] = ds
        _uri = f'{root_uri}#{groups[i]}'
        _save_dataset(ds, _uri, name)

    # Load and check data
    for i, (name, dataset) in enumerate(datasets.items()):
        _uri = f'{root_uri}#{groups[i]}'
        ds = _load_dataset(_uri)
        assert np.array_equal(dataset, ds[name].values)


def test_save_datasets_nested_zarr(uri, data):
    '''Test saving nested zarr files.'''
    datasets = {}
    root_uri = uri.rsplit('#', 1)[0]
    uris = [root_uri, f"{root_uri}/group.zarr"]

    for i, _uri in enumerate(uris):
        name = f'array{i}'
        ds = data.copy() * (i + 1)  # Make each dataset a bit different for testing
        datasets[name] = ds
        _save_dataset(ds, _uri, name)

    # Load and check data
    for i, (name, dataset) in enumerate(datasets.items()):
        # load each zarr directly
        _uri = uris[i]
        ds = _load_dataset(_uri)
        assert np.array_equal(dataset, ds[name].values)

        # Load nested zarr as group of root zarr, with consolidated = False
        store = ZarrIO().get_root(_uri)
        group = _uri.split(root_uri)[0]
        if group:
            ds = xr.open_zarr(store=store, group=group, consolidated=False)
            assert np.array_equal(dataset, ds[name].values)


def test_rename_dataset_dim_nocoords(uri, data):
    """Test rename dimension in zarr dataset with no coords."""
    _save_dataset(data, uri, "data")
    ds_a = ZarrIO().load_dataset(uri)
    rename_dict = {"dim_0": "x", "dim_1": "y"}
    for old, new in rename_dict.items():
        replace_dataset_dim(uri, old, new)

    ds_b = ZarrIO().load_dataset(uri)
    assert ds_a.rename(rename_dict).equals(ds_b)


def test_replace_dataset_dim_nocoords(uri, data):
    """Test replace dimension in zarr dataset with no coords."""
    _save_dataset(data, uri, "data")
    ds_a = ZarrIO().load_dataset(uri)
    with pytest.raises(ValueError):
        new_dim = xr.IndexVariable("x", ds_a["dim_0"].data)
        replace_dataset_dim(uri, "dim_0", new_dim)


def test_rename_dataset_dim(uri, tmp_3d_zarr):
    """Test rename dimension in zarr dataset."""
    ds_a = ZarrIO().load_dataset(tmp_3d_zarr)
    replace_dataset_dim(tmp_3d_zarr, "band", "abc")
    ds_b = ZarrIO().load_dataset(tmp_3d_zarr)
    assert ds_a.rename({"band": "abc"}).equals(ds_b)


@pytest.mark.parametrize("dtype", [np.int64, np.float64])
def test_replace_dataset_dim(tmp_3d_zarr, dtype):
    """Test replace dimension in zarr dataset."""
    ds_a = ZarrIO().load_dataset(tmp_3d_zarr)
    dim_len = len(ds_a["band"])
    new_name = "lambda"
    new_dim = xr.IndexVariable(new_name, np.linspace(123, 456, dim_len).astype(dtype))
    replace_dataset_dim(tmp_3d_zarr, "band", new_dim)
    ds_b = ZarrIO().load_dataset(tmp_3d_zarr)
    assert ds_b.variables[new_name].equals(new_dim)
    ds_a_new = ds_a.rename({"band": new_name}).assign_coords({new_name: new_dim})
    assert ds_a_new.equals(ds_b)


@pytest.mark.parametrize(
    "old,new", [("abc", "lambda"), ("array", "lambda"), ("band", "x")]
)
def test_replace_dataset_dim_badname(tmp_3d_zarr, old, new):
    """Test replacing zarr dimension with invalid names."""
    with pytest.raises(KeyError):
        replace_dataset_dim(tmp_3d_zarr, old, new)


def test_replace_dataset_dim_wronglen(tmp_3d_zarr):
    """Test replacing zarr dimension with bad length."""
    ds_a = ZarrIO().load_dataset(tmp_3d_zarr)
    new_dim = xr.IndexVariable("band2", np.append(ds_a["band"].data, [999]))
    with pytest.raises(ValueError):
        replace_dataset_dim(tmp_3d_zarr, "band", new_dim)
