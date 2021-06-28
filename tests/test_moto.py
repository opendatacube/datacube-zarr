"""Quick tests that each required library can access the moto s3 test flassk server."""

from pathlib import Path

import pytest
import fsspec
import rasterio
from s3path import S3Path

example_file = (
    Path(__file__).parent
    / "data/espa/ls8_sr/LC08_L1TP_091084_20190205_20190221_01_T1_sr_band1.tif"
)


@pytest.fixture(scope="session")
def tmp_s3_file_url(moto_s3_resource, s3):
    bucket = str(s3).split("/")[1]
    key = f"tmp/example/{example_file.name}"
    moto_s3_resource.Bucket(bucket).upload_file(str(example_file), key)
    yield f"s3://{bucket}/{key}"
    moto_s3_resource.Object(bucket, key).delete()


def test_mock_s3_fsspec(tmp_s3_file_url):
    prefix, filename = tmp_s3_file_url.rsplit("/", 1)
    fsmap = fsspec.get_mapper(prefix)
    assert filename in fsmap


def test_mock_s3_s3path(tmp_s3_file_url):
    s3file = S3Path.from_uri(tmp_s3_file_url)
    assert s3file.exists()


def test_mock_s3_gdal(tmp_s3_file_url):
    with rasterio.open(tmp_s3_file_url, "r") as src:
        assert src.shape == (336, 400)
