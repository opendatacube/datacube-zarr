'''Unit tests for the datacube_zarr.zarr_io module.'''

import pytest
import numpy as np
import xarray as xr

from datacube_zarr.zarr_io import ZarrIO, replace_dataset_dim

from .utils import _check_zarr_files, _load_dataset, _save_dataarray, _save_dataset


def test_consolidated_metadata_exists(tmp_3d_zarr):
    """Test that s3fs cache is working."""
    root = ZarrIO().get_root(tmp_3d_zarr)
    is_consolidated = ".zmetadata" in root
    assert is_consolidated


@pytest.mark.parametrize("save_fn", [_save_dataarray, _save_dataset])
def test_save_dataarray(tmp_storage_path, chunks, data, save_fn):
    '''Test ZarrIO.save_dataarray to save and load for a single DataArray.'''
    root = tmp_storage_path / "data.zarr"
    group = "dataset1"
    name = 'array1'
    uri = f"{root.as_uri()}#{group}"
    save_fn(data, uri, name, chunks['input'])

    # Check filesystem or s3 structure and some metadata
    _check_zarr_files(data, root, group, name, chunks)

    # Load and check data
    ds = _load_dataset(uri)
    assert np.array_equal(data, ds[name].values)


@pytest.mark.parametrize("save_fn", [_save_dataarray, _save_dataset])
def test_save_dataarrays(tmp_storage_path, chunks, data, save_fn):
    '''Test ZarrIO.save_dataarray to save and load multiple DataArrays.'''
    datasets = []
    root = tmp_storage_path / "data.zarr"
    for set_no in range(1, 3):
        group = f"dataset{set_no}"
        uri = f"{root.as_uri()}#{group}"
        name = f'array{set_no}'
        ds = data.copy() * set_no  # Make each dataset a bit different for testing
        datasets.append((uri, name, ds))
        save_fn(ds, uri, name, chunks['input'])

    assert len(datasets) == 2

    # Load and check data
    for uri, name, ds in datasets:
        ds_loaded = _load_dataset(uri)
        assert np.array_equal(ds, ds_loaded[name].values)


def test_print_tree(tmp_storage_path, fixed_chunks, data):
    '''Test zarr print data tree with a mix of Datasets and DataArrays co-existing.'''
    root = tmp_storage_path / "data.zarr"
    root_uri = root.as_uri()
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


def test_clean_store(tmp_storage_path, fixed_chunks, data):
    '''Test cleaning of zarr store.'''
    root = tmp_storage_path / "data.zarr"
    root_uri = root.as_uri()
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


def test_invalid_mode(example_uri, fixed_chunks, data):
    '''Test exceptions when an invalid mode is used.'''
    zio = ZarrIO()
    with pytest.raises(ValueError) as excinfo:
        zio.save_dataset(
            example_uri,
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
            example_uri,
            dataarray=data,
            name='array1',
            chunks=fixed_chunks['input'],
            mode='xxx',
        )
    assert (
        str(excinfo.value)
        == f"Only the following modes are supported {ZarrIO.WRITE_MODES}"
    )


def test_overwrite_dataset(example_uri, fixed_chunks, data):
    '''Test overwriting an existing dataset.'''
    zio = ZarrIO()
    # write the dataset twice
    for i in range(2):
        name = f'array{i}'
        dataset = (data.copy() + i).to_dataset(name=name)
        zio.save_dataset(
            uri=example_uri, dataset=dataset, chunks=fixed_chunks['input'], mode='w'
        )
    ds = zio.load_dataset(uri=example_uri)
    np.array_equal(dataset[name], ds[name].values)


def test_save_datasets_nested(tmp_storage_path, data):
    '''Test saving nested datasets (i.e. datasets and groups side by side).'''
    root_uri = (tmp_storage_path / "data.zarr").as_uri()
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


def test_save_datasets_nested_zarr(tmp_storage_path, data):
    '''Test saving nested zarr files.'''
    datasets = {}
    root_uri = (tmp_storage_path / "data.zarr").as_uri()
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
            ds = xr.open_dataset(
                store, group=group, engine="zarr", backend_kwargs={"consolidated": False}
            )
            assert np.array_equal(dataset, ds[name].values)


def test_rename_dataset_dim_nocoords(example_uri, data):
    """Test rename dimension in zarr dataset with no coords."""
    _save_dataset(data, example_uri, "data")
    ds_a = ZarrIO().load_dataset(example_uri)
    rename_dict = {"dim_0": "x", "dim_1": "y"}
    for old, new in rename_dict.items():
        replace_dataset_dim(example_uri, old, new)

    ds_b = ZarrIO().load_dataset(example_uri)
    assert ds_a.rename(rename_dict).equals(ds_b)


def test_replace_dataset_dim_nocoords(example_uri, data):
    """Test replace dimension in zarr dataset with no coords."""
    _save_dataset(data, example_uri, "data")
    ds_a = ZarrIO().load_dataset(example_uri)
    with pytest.raises(ValueError):
        new_dim = xr.IndexVariable("x", ds_a["dim_0"].data)
        replace_dataset_dim(example_uri, "dim_0", new_dim)


def test_rename_dataset_dim(tmp_3d_zarr):
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
