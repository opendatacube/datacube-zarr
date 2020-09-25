import random
import string
import threading
from pathlib import Path
from time import sleep
from types import SimpleNamespace

import pytest
import boto3
import numpy as np
import pyproj
import xarray as xr
from click.testing import CliRunner
from datacube import Datacube
from datacube.testutils import gen_tiff_dataset, mk_sample_dataset, mk_test_image
from moto import mock_s3
from moto.server import main as moto_server_main
from rasterio.crs import CRS
from s3path import S3Path, _s3_accessor

from datacube_zarr import ZarrIO
from datacube_zarr.tools.zarrify import main as zarrify
from datacube_zarr.utils.raster import raster_to_zarr
from datacube_zarr.utils.uris import uri_join
from tests.utils import copytree, create_random_raster

PROJECT_ROOT = Path(__file__).parents[1]

TEST_DATA = PROJECT_ROOT / 'tests' / 'data' / 'lbg'
TEST_DATA_LS8 = PROJECT_ROOT / 'tests' / 'data' / 'espa' / 'ls8_sr'

CHUNKS = (
    {  # When no chunk set, auto chunk for target 20MB. For a 1300x1300 data, it is:
        'input': None,
        'chunks_per_side': 1,
        'output': [1300, 1300],
    },
    {  # User specified chunks, input and output should match
        'input': {'dim_0': 1000, 'dim_1': 1100},
        'chunks_per_side': 2,
        'output': [1000, 1100],
    },
)
'''Zarr chunk sizes to be tested and expected output in metadata and number of chunks
per side.'''

s3_count = 0
'''Give a new ID to each moto bucket as they don't seem to clean properly between
runs.'''


_MOCK_S3_REGION = "mock-region"


@pytest.fixture
def s3_bucket_name():
    global s3_count
    yield f'mock-bucket-integration-{s3_count}'
    s3_count += 1


@pytest.fixture(scope='session')
def monkeypatch_session():
    """A patch for a session-scoped `monkeypatch`
    https://github.com/pytest-dev/pytest/issues/1872
    note: private import _pytest).
    """
    from _pytest.monkeypatch import MonkeyPatch

    m = MonkeyPatch()
    yield m
    m.undo()


@pytest.fixture(scope='session')
def mock_aws_aws_credentials(monkeypatch_session):
    '''Mocked AWS Credentials for moto.'''
    monkeypatch_session.setenv('AWS_ACCESS_KEY_ID', 'mock-key-id')
    monkeypatch_session.setenv('AWS_SECRET_ACCESS_KEY', 'mock-secret')
    monkeypatch_session.setenv('AWS_DEFAULT_REGION', _MOCK_S3_REGION)


@pytest.fixture(scope="session")
def moto_s3_server(monkeypatch_session):
    """Mock AWS S3 Server."""
    address = "http://127.0.0.1:5000"

    # GDAL AWS connection options
    monkeypatch_session.setenv('AWS_S3_ENDPOINT', address.split("://")[1])
    monkeypatch_session.setenv('AWS_VIRTUAL_HOSTING', 'FALSE')
    monkeypatch_session.setenv('AWS_HTTPS', 'NO')

    # Run a moto server
    thread = threading.Thread(target=moto_server_main, args=(["s3"],))
    thread.daemon = True
    thread.start()
    sleep(0.3)
    yield address


@pytest.fixture(scope="session")
def s3_client(moto_s3_server, mock_aws_aws_credentials):
    '''Mock s3 client.'''
    with mock_s3():
        client = boto3.client('s3', region_name=_MOCK_S3_REGION)
        _s3_accessor.s3 = boto3.resource('s3', region_name=_MOCK_S3_REGION)
        yield client


@pytest.fixture
def s3(s3_client, s3_bucket_name):
    '''Mock s3 client and root url.'''
    s3_client.create_bucket(
        Bucket=s3_bucket_name,
        CreateBucketConfiguration={'LocationConstraint': _MOCK_S3_REGION},
    )
    root = f'{s3_bucket_name}/mock-dir/mock-subdir'
    yield {'client': s3_client, 'root': root}


@pytest.fixture(params=('file', 's3'))
def uri(request, tmpdir, s3):
    '''Test URI parametrised for `file` and `s3` protocols.'''
    protocol = request.param
    root = s3['root'] if protocol == 's3' else Path(tmpdir) / 'data.zarr'
    group_name = 'dataset1'
    yield uri_join(protocol, root, group_name)


@pytest.fixture
def fixed_chunks():
    '''Single chunks.'''
    yield CHUNKS[1]


@pytest.fixture(params=CHUNKS)
def chunks(request):
    '''Parametrized chunks.'''
    yield request.param


@pytest.fixture
def data():
    '''Random test data.'''
    yield xr.DataArray(np.random.randn(1300, 1300))


@pytest.fixture
def dataset(tmpdir):
    """Datacube Dataset with random data.

    Based on datacube-core/tests/test_load_data.py"""
    tmpdir = Path(str(tmpdir))

    spatial = dict(resolution=(15, -15), offset=(11230, 1381110))

    nodata = -999
    array = mk_test_image(96, 64, 'int16', nodata=nodata)

    ds, gbox = gen_tiff_dataset(
        [SimpleNamespace(name='aa', values=array, nodata=nodata)],
        tmpdir,
        prefix='ds1-',
        timestamp='2018-07-19',
        **spatial,
    )
    sources = Datacube.group_datasets([ds], 'time')
    mm = ['aa']
    mm = [ds.type.measurements[k] for k in mm]
    dc_dataset = Datacube.load_data(sources, gbox, mm)

    # Flattening atributes: Zarr doesn't allow dicts
    for var_name in dc_dataset.data_vars:
        data_var = dc_dataset.data_vars[var_name]
        if 'spectral_definition' in data_var.attrs:
            spectral_definition = data_var.attrs.pop('spectral_definition', None)
            data_var.attrs['dc_spectral_definition_response'] = spectral_definition[
                'response'
            ]
            data_var.attrs['dc_spectral_definition_wavelength'] = spectral_definition[
                'wavelength'
            ]

    # Renaming units: units is a reserved name in Xarray coordinates
    for var_name in dc_dataset.coords:
        coord_var = dc_dataset.coords[var_name]
        if 'units' in coord_var.attrs:
            units = coord_var.attrs.pop('units', None)
            coord_var.attrs['dc_units'] = units

    yield dc_dataset


def _gen_zarr_dataset(ds, root):
    """Test dataset as loaded from zarr data in files.

    It comprises data attributes required in ODC."""
    var = list(ds.keys())[0]
    protocol = 'file'
    uri = uri_join(protocol, root)
    zio = ZarrIO()
    zio.save_dataset(uri=uri, dataset=ds)
    bands = [{'name': var, 'path': str(root)}]
    ds1 = mk_sample_dataset(bands, 'file', format='zarr')
    return ds1


@pytest.fixture
def odc_dataset(dataset, tmpdir):
    '''ODC test zarr dataset.'''
    root = Path(tmpdir) / 'data.zarr'
    yield _gen_zarr_dataset(dataset, root)


@pytest.fixture
def odc_dataset_2d(dataset, tmpdir):
    '''ODC test zarr dataset with only 2 dimensions.'''
    root = Path(tmpdir) / 'data_2d.zarr'
    dataset = dataset.squeeze(drop=True)
    yield _gen_zarr_dataset(dataset, root)


@pytest.fixture(params=('file', 's3'))
def tmp_storage_path(request, tmp_path, s3):
    """Temporary storage path."""
    protocol = request.param
    prefix = ''.join(random.choices(string.ascii_letters + string.digits, k=6))
    if protocol == "s3":
        return S3Path(f"/{s3['root']}/{prefix}/")
    else:
        return tmp_path / prefix


tmp_raster_storage_path = tmp_storage_path


@pytest.fixture()
def tmp_raster(tmp_raster_storage_path):
    """Temporary geotif."""
    outdir = tmp_raster_storage_path / "geotif"
    raster = create_random_raster(outdir)
    yield raster


@pytest.fixture()
def tmp_raster_multiband(tmp_raster_storage_path):
    """Temporary multiband geotif."""
    outdir = tmp_raster_storage_path / "geotif_multi"
    raster = create_random_raster(outdir, nbands=5)
    yield raster


@pytest.fixture
def tmp_3d_zarr(tmp_raster_multiband):
    out_dir = tmp_raster_multiband.parent
    chunks = {"x": 100, "y": 100, "band": -1}
    uris = raster_to_zarr(
        tmp_raster_multiband, out_dir=out_dir, multi_dim=True, chunks=chunks
    )
    assert len(uris) == 1
    yield uris[0]


@pytest.fixture()
def tmp_hdf4_dataset(tmp_path):
    """Create a HDF4 path."""
    outdir = tmp_path / "geotif"
    outdir.mkdir()
    raster = create_random_raster(outdir, nbands=5)
    da = xr.open_rasterio(raster.as_uri())
    crs = CRS.from_string(da.crs).to_string()

    # make dataset and add spatial ref
    grid_map_attrs = pyproj.CRS.from_string(crs).to_cf()
    da.coords["spatial_ref"] = xr.Variable((), 0)
    da.coords["spatial_ref"].attrs.update(grid_map_attrs)
    da.attrs["grid_mapping"] = "spatial_ref"
    ds = da.to_dataset(dim="band")
    ds = ds.rename_vars({k: f"band{k}" for k in ds.data_vars.keys()})
    for var in ds.data_vars:
        ds[var].attrs["grid_mapping"] = "spatial_ref"

    # save as netcdf
    hdf_path = tmp_path / "hdf" / f"{raster.stem}.nc"
    hdf_path.parent.mkdir()
    ds.to_netcdf(hdf_path, format="NETCDF4")

    return hdf_path


@pytest.fixture()
def tmp_dir_of_rasters(tmp_raster_storage_path):
    """Temporary directory of geotifs."""
    outdir = tmp_raster_storage_path / "geotif_scene"
    rasters = [create_random_raster(outdir, label=f"raster{i}") for i in range(5)]
    others = [outdir / "metadata.xml", outdir / "path" / "to" / "otherfile.txt"]
    for o in others:
        o.parent.mkdir(exist_ok=True, parents=True)
        o.touch()
    yield outdir, rasters, others


@pytest.fixture()
def tmp_empty_dataset(tmp_path):
    """Empty geotif."""
    hdf = tmp_path / "empty" / "nothing.hdf"
    hdf.parent.mkdir(parents=True)
    xr.DataArray().to_netcdf(hdf)
    yield hdf


@pytest.fixture(params=["file", "s3"])
def ls5_dataset_path(request, s3, tmp_path):
    """LS5 test dataset on filesystem and s3."""
    if request.param == "file":
        dataset_path = tmp_path / "geotifs" / "lbg"
    else:
        bucket, root = s3["root"].split("/", 1)
        dataset_path = S3Path(f"/{bucket}/{root}/geotifs/lbg")
    copytree(TEST_DATA, dataset_path)
    return dataset_path


@pytest.fixture(params=["file", "s3"])
def ls8_dataset_path(request, s3, tmp_path):
    """LS8 test dataset on filesystem and s3."""
    if request.param == "file":
        dataset_path = tmp_path / "geotifs" / "espa" / "ls8_sr"
    else:
        bucket, root = s3["root"].split("/", 1)
        dataset_path = S3Path(f"/{bucket}/{root}/geotifs/lbg")
    copytree(TEST_DATA_LS8, dataset_path)
    return dataset_path


@pytest.fixture(scope="session")
def zarrifycli():
    """zarrify runner."""
    runner = CliRunner()

    def _run(args):
        res = runner.invoke(zarrify, args)
        return res

    return _run
