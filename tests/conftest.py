import multiprocessing
import re
from pathlib import Path
from time import sleep
from types import SimpleNamespace

import pytest
import boto3
import fsspec
import numpy as np
import pyproj
import xarray as xr
from click.testing import CliRunner
from datacube import Datacube
from datacube.testutils import gen_tiff_dataset, mk_sample_dataset, mk_test_image
from moto.server import main as moto_server_main
from rasterio.crs import CRS
from s3path import S3Path, register_configuration_parameter

from datacube_zarr import ZarrIO
from datacube_zarr.tools.set_zarr_product_extra_dim import (
    cli as set_zarr_product_extra_dim,
)
from datacube_zarr.tools.zarrify import cli as zarrify
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
def moto_aws_credentials(monkeypatch_session):
    '''Mocked AWS Credentials for moto.'''

    MOCK_AWS_CREDENTIALS = {
        'AWS_ACCESS_KEY_ID': 'mock-key-id',
        'AWS_SECRET_ACCESS_KEY': 'mock-secret',
        'AWS_DEFAULT_REGION': "mock-region",
    }
    for k, v in MOCK_AWS_CREDENTIALS.items():
        monkeypatch_session.setenv(k, v)

    return MOCK_AWS_CREDENTIALS


@pytest.fixture(scope="session")
def moto_s3_server(monkeypatch_session):
    """Mock AWS S3 Server."""
    address = "http://127.0.0.1:5000"

    # Run a moto server
    proc = multiprocessing.Process(
        target=moto_server_main,
        name="moto_s3_server",
        args=(["s3"],),
        daemon=True,
    )
    proc.start()
    sleep(0.3)
    yield address
    proc.terminate()
    proc.join()


@pytest.fixture(scope='session')
def gdal_mock_s3_endpoint(moto_s3_server, monkeypatch_session):
    """Set environment variables for GDAL/rasterio to access moto server."""
    monkeypatch_session.setenv('AWS_S3_ENDPOINT', moto_s3_server.split("://")[1])
    monkeypatch_session.setenv('AWS_VIRTUAL_HOSTING', 'FALSE')
    monkeypatch_session.setenv('AWS_HTTPS', 'NO')


@pytest.fixture(scope='session')
def fsspec_mock_s3_endpoint(moto_s3_server, moto_aws_credentials):
    """Set the boto s3 endpoint via fspec config.

    Boto libraries don't offer any way to do this."""

    fsspec_conf = {
        "s3": {
            "client_kwargs": {
                "endpoint_url": moto_s3_server,
                "region_name": moto_aws_credentials['AWS_DEFAULT_REGION'],
            }
        }
    }
    fsspec.config.conf.update(fsspec_conf)


@pytest.fixture(scope="session")
def moto_s3_resource(moto_s3_server, moto_aws_credentials):
    """A boto3 s3 resource pointing to the moto server."""
    s3resource = boto3.resource(
        's3',
        endpoint_url=moto_s3_server,
        aws_access_key_id=moto_aws_credentials['AWS_ACCESS_KEY_ID'],
        aws_secret_access_key=moto_aws_credentials['AWS_SECRET_ACCESS_KEY'],
        # config=Config(signature_version='s3v4'),
        region_name=moto_aws_credentials['AWS_DEFAULT_REGION'],
    )
    return s3resource


@pytest.fixture(scope="session")
def s3path_mock_s3_endpoint(moto_s3_resource):
    """Set boto resource for s3path libaray to access moto server."""
    all_buckets = S3Path('/')
    register_configuration_parameter(all_buckets, resource=moto_s3_resource)


@pytest.fixture(scope="session")
def s3(
    gdal_mock_s3_endpoint,
    s3path_mock_s3_endpoint,
    fsspec_mock_s3_endpoint,
):
    """Collect together all requires per-session mock s3 fixtures and return a bucket."""
    s3_bucket = S3Path("/mock-s3-bucket")
    s3_bucket.mkdir()
    return s3_bucket


@pytest.fixture(scope="session")
def tmp_s3path_factory(s3):
    """S3Path version of pytest tmp_path_factory."""

    def _as_int(s):
        try:
            return int(s)
        except ValueError:
            return -1

    class TmpS3PathFactory:
        def __init__(self, basetmp):
            self.basetmp = basetmp

        def mktemp(self, name):
            suffixes = [
                str(p.relative_to(self.basetmp))[len(name) :]
                for p in self.basetmp.glob(f"{name}*")
            ]
            max_existing = max([_as_int(s) for s in suffixes], default=-1)
            p = self.basetmp / f"{name}{max_existing + 1}"
            return p

    return TmpS3PathFactory(basetmp=s3 / "pytest")


@pytest.fixture()
def tmp_s3path(request, tmp_s3path_factory):
    """S3Path vesrion of tmp_path fixture."""
    MAXVAL = 30
    name = re.sub(r"[\W]", "_", request.node.name)[:MAXVAL]
    return tmp_s3path_factory.mktemp(name)


@pytest.fixture(params=('file', 's3'))
def tmp_storage_path(request, tmp_path, tmp_s3path):
    return tmp_s3path if request.param == "s3" else tmp_path


@pytest.fixture()
def tmp_input_storage_path(tmp_storage_path):
    return tmp_storage_path / "input"


@pytest.fixture()
def tmp_output_storage_path(tmp_storage_path):
    return tmp_storage_path / "output"


@pytest.fixture()
def example_uri(tmp_storage_path):
    '''Test URI parametrised for `file` and `s3` protocols.'''
    return (tmp_storage_path / "data.zarr").as_uri() + "#dataset1"


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
def dataset(tmp_path):
    """Datacube Dataset with random data.

    Based on datacube-core/tests/test_load_data.py"""

    spatial = dict(resolution=(15, -15), offset=(11230, 1381110))

    nodata = -999
    array = mk_test_image(96, 64, 'int16', nodata=nodata)

    ds, gbox = gen_tiff_dataset(
        [SimpleNamespace(name='aa', values=array, nodata=nodata)],
        tmp_path,
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


def _gen_zarr_dataset(ds, root, group=None):
    """Test dataset as loaded from zarr data in files.

    It comprises data attributes required in ODC."""
    var = list(ds.keys())[0]
    protocol = 'file'
    uri = uri_join(protocol, root, group)
    zio = ZarrIO()
    zio.save_dataset(uri=uri, dataset=ds)
    bands = [{'name': var, 'path': str(root)}]
    ds1 = mk_sample_dataset(bands, 'file', format='zarr')
    return ds1


@pytest.fixture
def zarr_with_group(dataset, tmp_path):
    '''ODC test zarr dataset.'''
    root = tmp_path / 'zarr_with_group.zarr'
    uri = uri_join("file", root, group="group")
    zio = ZarrIO()
    zio.save_dataset(uri=uri, dataset=dataset)
    yield uri


@pytest.fixture
def odc_dataset(dataset, tmp_path):
    '''ODC test zarr dataset.'''
    root = tmp_path / 'data.zarr'
    yield _gen_zarr_dataset(dataset, root)


@pytest.fixture
def odc_dataset_2d(dataset, tmp_path):
    '''ODC test zarr dataset with only 2 dimensions.'''
    root = tmp_path / 'data_2d.zarr'
    dataset = dataset.squeeze(drop=True)
    yield _gen_zarr_dataset(dataset, root)


@pytest.fixture()
def tmp_raster(tmp_input_storage_path):
    """Temporary geotif."""
    d = tmp_input_storage_path / "geotif"
    raster = create_random_raster(d)
    yield raster


@pytest.fixture()
def tmp_raster_multiband(tmp_input_storage_path):
    """Temporary multiband geotif."""
    d = tmp_input_storage_path / "geotif_multi"
    raster = create_random_raster(d, nbands=5)
    yield raster


@pytest.fixture
def tmp_3d_zarr(tmp_raster_multiband):
    out_dir = tmp_raster_multiband.parent
    chunks = {"x": 100, "y": 100, "band": -1}
    uris = raster_to_zarr(tmp_raster_multiband, out_dir=out_dir, chunks=chunks)
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
def tmp_dir_of_rasters(tmp_input_storage_path):
    """Temporary directory of geotifs."""
    outdir = tmp_input_storage_path / "geotif_scene"
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


@pytest.fixture()
def ls5_dataset_path(tmp_input_storage_path):
    """LS5 test dataset on filesystem and s3."""
    dataset_path = tmp_input_storage_path / "geotifs" / "lbg"
    copytree(TEST_DATA, dataset_path)
    return dataset_path


@pytest.fixture()
def ls8_dataset_path(tmp_input_storage_path):
    """LS8 test dataset on filesystem and s3."""
    dataset_path = tmp_input_storage_path / "geotifs" / "lbg"
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


@pytest.fixture(scope="session")
def set_zarr_product_extra_dim_cli():
    """set_zarr_product_extra_dim runner."""
    runner = CliRunner()

    def _run(args):
        res = runner.invoke(set_zarr_product_extra_dim, args)
        return res

    return _run
