import pytest
import boto3
import botocore
from s3path import S3Path

from datacube_zarr.utils.convert import convert_dir, convert_to_zarr, get_datasets
from datacube_zarr.utils.uris import uri_split

from .utils import _load_dataset, copytree, raster_and_zarr_are_equal


def test_mock_s3_path(s3):
    """Check test bucket exists with S3Path."""
    bucket = s3["root"].split("/")[0]
    path = S3Path(f"/{bucket}")
    assert path.exists()


def test_mock_s3_botocore(s3):
    """Check test bucket exists with botocore."""
    bucket = s3["root"].split("/")[0]
    client = botocore.session.Session().create_client("s3")
    resp = client.head_bucket(Bucket=bucket)
    assert resp["ResponseMetadata"]["HTTPStatusCode"] == 200


def test_mock_s3_boto3_client(s3):
    """Check test bucket exists with boto3 client."""
    bucket = s3["root"].split("/")[0]
    client = boto3.client("s3")
    resp = client.head_bucket(Bucket=bucket)
    assert resp["ResponseMetadata"]["HTTPStatusCode"] == 200


def test_mock_s3_boto3_resource(s3):
    """Check test bucket exists with boto3 resource."""
    bucket = s3["root"].split("/")[0]
    s3 = boto3.resource("s3")
    assert s3.Bucket(bucket) in s3.buckets.all()


def test_find_datasets_geotif(tmp_dir_of_rasters):
    """Test finding geotif datasets."""
    data_dir, geotifs, others = tmp_dir_of_rasters
    found_types, found_datasets = zip(*get_datasets(data_dir))
    found_geotifs = [ds[0] for ds in found_datasets]
    assert all(t == "GeoTiff" for t in found_types)
    assert set(geotifs) == set(found_geotifs)


@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("merge_datasets_per_dir", [False, True])
def test_convert_dir_geotif(
    tmp_dir_of_rasters, tmp_storage_path, merge_datasets_per_dir, inplace
):
    """Test converting a directory of geotifs."""
    data_dir, geotifs, others = tmp_dir_of_rasters
    others_rel = [o.relative_to(data_dir) for o in others]
    if inplace:
        copytree(data_dir, tmp_storage_path)
        data_dir = tmp_storage_path
        outdir = None
    else:
        outdir = tmp_storage_path / "outdir"

    zarrs = convert_dir(data_dir, outdir, merge_datasets_per_dir=merge_datasets_per_dir)

    # check generated zarrs
    assert len(zarrs) == len(geotifs)
    for z, g in zip(sorted(zarrs), sorted(geotifs)):
        protocol, root, group = uri_split(z)
        root_stem = root.rsplit("/", 1)[1].split(".")[0]
        if merge_datasets_per_dir:
            assert root_stem == "raster"
            assert group == g.stem
        else:
            assert root_stem == g.stem
            assert group == ""

        assert raster_and_zarr_are_equal(g.as_uri(), z)

    # check other data
    converted_dir = outdir or data_dir
    for o in others_rel:
        assert (converted_dir / o).exists()


def test_convert_ls8(ls8_dataset_path, tmp_path):
    """Test converting ls8 dataset."""
    out_dir = tmp_path / "out_dir"
    zarrs = convert_dir(ls8_dataset_path, out_dir)
    for z in zarrs:
        ds = _load_dataset(z)
        assert ds["band1"].size > 0


def test_convert_unsupported(tmp_path):
    """Test unsupported file format."""
    dataset = tmp_path / "data.he5"
    dataset.touch()
    with pytest.raises(ValueError):
        convert_to_zarr([dataset])
