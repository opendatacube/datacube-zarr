import hashlib

import pytest
import rasterio
from affine import Affine

from datacube.api.query import query_group_by
from datacube.utils import geometry
from integration_tests.test_end_to_end import INGESTER_CONFIGS
from integration_tests.utils import GEOTIFF, prepare_test_ingestion_configuration

EXPECTED_STORAGE_UNIT_DATA_SHAPE = (1, 40, 40)
COMPLIANCE_CHECKER_NORMAL_LIMIT = 2


@pytest.mark.timeout(20)
@pytest.mark.parametrize('datacube_env_name', ('datacube',), indirect=True)
@pytest.mark.usefixtures('default_metadata_type',
                         'indexed_ls5_scene_products',
                         's3')
def test_full_ingestion(clirunner, index, tmpdir, example_ls5_dataset_paths,
                        ingest_configs, s3_bucket_name):
    config = INGESTER_CONFIGS/ingest_configs['ls5_nbar_albers']
    config_path, config = prepare_test_ingestion_configuration(
        tmpdir, None, config, mode='fast_ingest', s3_bucket_name=s3_bucket_name)
    valid_uuids = []
    for uuid, example_ls5_dataset_path in example_ls5_dataset_paths.items():
        valid_uuids.append(uuid)
        clirunner([
            'dataset',
            'add',
            str(example_ls5_dataset_path)
        ])

    ensure_datasets_are_indexed(index, valid_uuids)

    clirunner([
        'ingest',
        '--config-file',
        str(config_path)
    ])

    datasets = index.datasets.search_eager(product='ls5_nbar_albers')
    assert len(datasets) > 0
    assert datasets[0].managed

    check_open_with_api(index, len(valid_uuids))
    check_data_with_api(index, len(valid_uuids))


@pytest.mark.timeout(20)
@pytest.mark.parametrize('datacube_env_name', ('datacube',), indirect=True)
@pytest.mark.usefixtures('default_metadata_type',
                         'indexed_ls5_scene_products',
                         's3')
def test_process_all_ingest_jobs(clirunner, index, tmpdir, example_ls5_dataset_paths,
                                 ingest_configs, s3_bucket_name):
    """
    Test for the case where ingestor processes upto `--queue-size` number of tasks and
    not all the available scenes
    """
    # Make a test ingestor configuration
    config = INGESTER_CONFIGS / ingest_configs['ls5_nbar_albers']
    config_path, config = prepare_test_ingestion_configuration(
        tmpdir, None, config, mode='fast_ingest', s3_bucket_name=s3_bucket_name)

    def index_dataset(path):
        return clirunner(['dataset', 'add', str(path)])

    # Number of scenes generated is 3 (as per NUM_TIME_SLICES const from conftest.py)
    # Set the queue size to process 2 tiles
    queue_size = 2
    valid_uuids = []
    for uuid, ls5_dataset_path in example_ls5_dataset_paths.items():
        valid_uuids.append(uuid)
        index_dataset(ls5_dataset_path)

    # Ensure that datasets are actually indexed
    ensure_datasets_are_indexed(index, valid_uuids)

    # Ingest all scenes (Though the queue size is 2, all 3 tiles will be ingested)
    clirunner([
        'ingest',
        '--config-file',
        str(config_path),
        '--queue-size',
        queue_size,
        '--allow-product-changes',
    ])

    # Validate that the ingestion is working as expected
    datasets = index.datasets.search_eager(product='ls5_nbar_albers')
    assert len(datasets) > 0
    assert datasets[0].managed

    check_open_with_api(index, len(valid_uuids))


def ensure_datasets_are_indexed(index, valid_uuids):
    datasets = index.datasets.search_eager(product='ls5_nbar_scene')
    assert len(datasets) == len(valid_uuids)
    for dataset in datasets:
        assert dataset.id in valid_uuids


def check_open_with_api(index, time_slices):
    with rasterio.Env():
        from datacube import Datacube
        dc = Datacube(index=index)

        input_type_name = 'ls5_nbar_albers'
        input_type = dc.index.products.get_by_name(input_type_name)
        geobox = geometry.GeoBox(200, 200, Affine(25, 0.0, 638000, 0.0, -25, 6276000), geometry.CRS('EPSG:28355'))
        observations = dc.find_datasets(product='ls5_nbar_albers', geopolygon=geobox.extent)
        group_by = query_group_by('time')
        sources = dc.group_datasets(observations, group_by)
        data = dc.load_data(sources, geobox, input_type.measurements.values())
        assert data.blue.shape == (time_slices, 200, 200)

        chunk_profile = {'time': 1, 'x': 100, 'y': 100}
        lazy_data = dc.load_data(sources, geobox, input_type.measurements.values(), dask_chunks=chunk_profile)
        assert lazy_data.blue.shape == (time_slices, 200, 200)
        assert (lazy_data.blue.load() == data.blue).all()


def check_data_with_api(index, time_slices):
    """Chek retrieved data for specific values.

    We scale down by 100 and check for predefined values in the
    corners.
    """
    from datacube import Datacube
    dc = Datacube(index=index)

    # TODO: this test needs to change, it tests that results are exactly the
    #       same as some time before, but with the current zoom out factor it's
    #       hard to verify that results are as expected even with human
    #       judgement. What it should test is that reading native from the
    #       ingested product gives exactly the same results as reading into the
    #       same GeoBox from the original product. Separate to that there
    #       should be a read test that confirms that what you read from native
    #       product while changing projection is of expected value

    # Make the retrieved data lower res
    ss = 100
    shape_x = int(GEOTIFF['shape']['x'] / ss)
    shape_y = int(GEOTIFF['shape']['y'] / ss)
    pixel_x = int(GEOTIFF['pixel_size']['x'] * ss)
    pixel_y = int(GEOTIFF['pixel_size']['y'] * ss)

    input_type_name = 'ls5_nbar_albers'
    input_type = dc.index.products.get_by_name(input_type_name)
    geobox = geometry.GeoBox(shape_x + 2, shape_y + 2,
                             Affine(pixel_x, 0.0, GEOTIFF['ul']['x'], 0.0, pixel_y, GEOTIFF['ul']['y']),
                             geometry.CRS(GEOTIFF['crs']))
    observations = dc.find_datasets(product='ls5_nbar_albers', geopolygon=geobox.extent)
    group_by = query_group_by('time')
    sources = dc.group_datasets(observations, group_by)
    data = dc.load_data(sources, geobox, input_type.measurements.values())
    assert hashlib.md5(data.green.data).hexdigest() == '0f64647bad54db4389fb065b2128025e'
    assert hashlib.md5(data.blue.data).hexdigest() == '41a7b50dfe5c4c1a1befbc378225beeb'
    for time_slice in range(time_slices):
        assert data.blue.values[time_slice][-1, -1] == -999
