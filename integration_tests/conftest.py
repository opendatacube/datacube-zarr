# coding=utf-8
"""
Common methods for index integration tests.
"""
import itertools
import multiprocessing
import os
import re
from pathlib import Path
from time import sleep

import pytest
import boto3
import datacube.scripts.cli_app
import datacube.utils
import fsspec
from click.testing import CliRunner
from datacube.config import LocalConfig
from datacube.drivers.postgres import PostgresDb, _core
from datacube.index import index_connect
from datacube.index._metadata_types import default_metadata_type_docs
from hypothesis import HealthCheck, settings
from moto.server import main as moto_server_main
from s3path import S3Path, register_configuration_parameter

from integration_tests.utils import copytree

INTEGRATION_TESTS_DIR = Path(__file__).parent

#: Number of time slices to create in sample data
NUM_TIME_SLICES = 3

PROJECT_ROOT = Path(__file__).parents[1]

TEST_DATA = PROJECT_ROOT / 'tests' / 'data' / 'lbg'
LBG_NBAR = 'LS5_TM_NBAR_P54_GANBAR01-002_090_084_19920323'
LBG_PQ = 'LS5_TM_PQ_P55_GAPQ01-002_090_084_19920323'

TEST_DATA_LS8 = PROJECT_ROOT / 'tests' / 'data' / 'espa' / 'ls8_sr'

CONFIG_SAMPLES = PROJECT_ROOT / 'docs' / 'config_samples'

CONFIG_FILE_PATHS = [
    str(INTEGRATION_TESTS_DIR / 'agdcintegration.conf'),
    os.path.expanduser('~/.datacube_integration.conf'),
]

# Configure Hypothesis to allow slower tests, because we're testing datasets
# and disk IO rather than scalar values in memory.  Ask @Zac-HD for details.
settings.register_profile(
    'opendatacube',
    deadline=5000,
    max_examples=10,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile('opendatacube')


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
    s3_bucket = S3Path("/mock-s3-bucket-integration")
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


@pytest.fixture(scope="session")
def global_integration_cli_args():
    """
    The first arguments to pass to a cli command for integration test configuration.
    """
    # List of a config files in order.
    return list(itertools.chain(*(('--config', f) for f in CONFIG_FILE_PATHS)))


@pytest.fixture
def datacube_env_name(request):
    if hasattr(request, 'param'):
        return request.param
    else:
        return 'datacube'


@pytest.fixture
def local_config(datacube_env_name):
    """Provides a :class:`LocalConfig` configured with suitable config file paths.

    .. seealso::

        The :func:`integration_config_paths` fixture sets up the config files.
    """
    return LocalConfig.find(CONFIG_FILE_PATHS, env=datacube_env_name)


@pytest.fixture
def uninitialised_postgres_db(local_config):
    """
    Return a connection to an empty PostgreSQL database
    """
    timezone = "UTC"
    db = PostgresDb.from_config(
        local_config, application_name='test-run', validate_connection=False
    )

    # Drop tables so our tests have a clean db.
    _core.drop_db(db._engine)
    db._engine.execute(
        'alter database %s set timezone = %r' % (local_config['db_database'], timezone)
    )

    # We need to run this as well
    # I think because SQLAlchemy grabs them into it's MetaData,
    # and attempts to recreate them. TODO FIX
    remove_dynamic_indexes()

    yield db
    # with db.begin() as c:  # Drop SCHEMA
    _core.drop_db(db._engine)
    db.close()


@pytest.fixture
def index(local_config, uninitialised_postgres_db: PostgresDb):
    index = index_connect(local_config, validate_connection=False)
    index.init_db()
    return index


@pytest.fixture
def index_empty(local_config, uninitialised_postgres_db: PostgresDb):
    index = index_connect(local_config, validate_connection=False)
    index.init_db(with_default_types=False)
    return index


def remove_dynamic_indexes():
    """
    Clear any dynamically created postgresql indexes from the schema.
    """
    # Our normal indexes start with "ix_", dynamic indexes with "dix_"
    for table in _core.METADATA.tables.values():
        table.indexes.intersection_update(
            [i for i in table.indexes if not i.name.startswith('dix_')]
        )


@pytest.fixture
def default_metadata_type_doc():
    return [doc for doc in default_metadata_type_docs() if doc['name'] == 'eo'][0]


@pytest.fixture
def default_metadata_types(index):
    """Inserts the default metadata types into the Index"""
    for d in default_metadata_type_docs():
        index.metadata_types.add(index.metadata_types.from_doc(d))
    return index.metadata_types.get_all()


@pytest.fixture
def default_metadata_type(index, default_metadata_types):
    return index.metadata_types.get_by_name('eo')


@pytest.fixture()
def ls5_dataset_path(tmp_input_storage_path):
    """LS5 test dataset on filesystem and s3."""
    dataset_path = tmp_input_storage_path / "geotifs" / "lbg"
    copytree(TEST_DATA, dataset_path)
    return dataset_path


@pytest.fixture()
def ls8_dataset_path(tmp_path):
    """LS8 test dataset on filesystem."""
    dataset_path = tmp_path / "geotifs" / "espa" / "ls8_sr"
    copytree(TEST_DATA_LS8, dataset_path)
    return dataset_path


@pytest.fixture
def clirunner(global_integration_cli_args, datacube_env_name):
    def _run_cli(
        opts,
        catch_exceptions=False,
        expect_success=True,
        cli_method=datacube.scripts.cli_app.cli,
        verbose_flag='-v',
    ):
        exe_opts = list(itertools.chain(*(('--config', f) for f in CONFIG_FILE_PATHS)))
        exe_opts += ['--env', datacube_env_name]
        if verbose_flag:
            exe_opts.append(verbose_flag)
        exe_opts.extend(opts)

        runner = CliRunner()
        result = runner.invoke(cli_method, exe_opts, catch_exceptions=catch_exceptions)
        if expect_success:
            assert 0 == result.exit_code, "Error for %r. output: %r" % (
                opts,
                result.output,
            )
        return result

    return _run_cli
