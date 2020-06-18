from pathlib import Path
from types import SimpleNamespace

import pytest
import boto3
from s3path import S3Path
import numpy as np
from datacube import Datacube
from datacube.testutils import gen_tiff_dataset, mk_sample_dataset, mk_test_image
from moto import mock_s3
import rasterio
from xarray import DataArray

from zarr_io.zarr_io import ZarrIO

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
'''Zarr chunk sizes to be tested and expected output in metadata and number of chunks
per side.'''

s3_count = 0
'''Give a new ID to each moto bucket as they don't seem to clean properly between
runs.'''


@pytest.fixture
def s3_bucket_name():
    global s3_count
    yield f'mock-bucket-{s3_count}'
    s3_count += 1


@pytest.fixture
def mock_aws_aws_credentials(monkeypatch):
    '''Mocked AWS Credentials for moto.'''
    monkeypatch.setenv('AWS_ACCESS_KEY_ID', 'mock-key-id')
    monkeypatch.setenv('AWS_SECRET_ACCESS_KEY', 'mock-secret')
    monkeypatch.setenv('AWS_DEFAULT_REGION', 'mock-region')


@pytest.fixture
def s3(s3_bucket_name, mock_aws_aws_credentials):
    '''Mock s3 client and root url.'''
    with mock_s3():
        client = boto3.client('s3', region_name='mock-region')
        client.create_bucket(Bucket=s3_bucket_name)
        root = f'{s3_bucket_name}/mock-dir/mock-subdir'
        yield {'client': client, 'root': root}


@pytest.fixture(params=('file', 's3'))
def uri(request, tmpdir, s3):
    '''Test URI parametrised for `file` and `s3` protocols.'''
    protocol = request.param
    root = s3['root'] if protocol == 's3' else Path(tmpdir) / 'data.zarr'
    group_name = 'dataset1'
    yield f'{protocol}://{root}#{group_name}'


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
    yield DataArray(np.random.randn(1300, 1300))


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

    yield dc_dataset


def _gen_zarr_dataset(ds, root):
    '''Test dataset as loaded from zarr data in files.

    It comprises data attributes required in ODC.'''
    var = list(ds.keys())[0]
    protocol = 'file'
    uri = f'{protocol}://{root}'
    zio = ZarrIO()
    zio.save_dataset(uri=uri, dataset=ds)
    bands = [{
        'name': var,
        'path': str(root)
    }]
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


def create_random_raster(
    outdir: Path,
    label: str = "raster",
    height: int = 200,
    width: int = 300,
    nbands: int = 1
) -> Path:
    """Create a raster with random data."""
    outdir.mkdir()
    dtype = np.float32
    data = np.random.randn(nbands, height, width).astype(dtype)
    file_path = outdir / f"{label}_{nbands}x{height}x{width}.tif"

    bbox = [149, 35, 150, 36]
    transform = rasterio.transform.from_bounds(*bbox, width, height)
    meta = {
        "driver": "GTiff",
        "count": nbands,
        "width": width,
        "height": height,
        "crs": rasterio.crs.CRS.from_epsg("4326"),
        "nodata": None,
        "dtype": dtype,
        "transform": transform,
    }

    with rasterio.open(file_path.as_uri(), "w", **meta) as dst:
        dst.write(data)

    return file_path


@pytest.fixture(params=('file', 's3'))
def tmp_storage_path(request, tmp_path, s3):
    """Temporary storage path."""
    protocol = request.param
    if protocol == "s3":
        return S3Path(f"/{s3['root']}/tmp/")
    else:
        return tmp_path


@pytest.fixture(scope="session")
def tmp_raster(tmp_storage_path):
    """Temporary geotif."""
    outdir = tmp_storage_path / "geotif"
    raster = create_random_raster(outdir)
    yield raster


@pytest.fixture(scope="session")
def tmp_raster_multiband(tmp_storage_path):
    """Temporary multiband geotif."""
    outdir = tmp_storage_path / "geotif_multi"
    raster = create_random_raster(outdir, nbands=5)
    yield raster


@pytest.fixture(scope="session")
def tmp_dir_of_rasters(tmp_storage_path):
    """Temporary directory of geotifs."""
    outdir = tmp_storage_path / "geotif_scene"
    rasters = [create_random_raster(outdir, label=f"raster{i}") for i in range(5)]
    yield outdir, rasters
