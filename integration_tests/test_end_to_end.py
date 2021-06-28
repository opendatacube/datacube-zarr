# Copied from OpenDataCube integration tests https://github.com/opendatacube/datacube-core
#
# End to end test of zariffied ls5 dataset converted to albers and indexed

import shutil
from pathlib import Path

import pytest
import numpy
import rasterio
import yaml
from click.testing import CliRunner
from datacube.api.core import Datacube

from datacube_zarr.tools.zarrify import cli as zarrify
from examples.prepare_zarr_ls5 import main as prepare_zarr_ls5

PROJECT_ROOT = Path(__file__).parents[1]
CONFIG_SAMPLES = PROJECT_ROOT / 'docs/config_samples/'
LS5_DATASET_TYPES = CONFIG_SAMPLES / 'dataset_types/ls5_scenes.yaml'
LS5_DATASET_TYPES_ZARR = CONFIG_SAMPLES / "dataset_types/ls5_scenes_albers_zarr.yaml"
TEST_DATA = PROJECT_ROOT / 'tests' / 'data' / 'lbg'
LBG_NBAR = 'LS5_TM_NBAR_P54_GANBAR01-002_090_084_19920323'
LBG_PQ = 'LS5_TM_PQ_P55_GAPQ01-002_090_084_19920323'
LBG_CELL = (15, -40)  # x,y


def custom_dumb_fuser(dst, src):
    dst[:] = src[:]


@pytest.fixture()
def testdata_dir(tmpdir):
    datadir = Path(str(tmpdir), 'data')
    datadir.mkdir()
    shutil.copytree(str(TEST_DATA), str(datadir / 'lbg'))
    return datadir


def replace_band_names(metadatafile):
    """The LS5 EO prepare script sets band by number but this product uses names."""
    band_map = {
        '1': 'blue',
        '2': 'green',
        '3': 'red',
        '4': 'nir',
        '5': 'swir1',
        '7': 'swir2',
    }
    meta = yaml.safe_load(metadatafile.read_text())

    def _replace_bands(band_meta):
        new_bands = {}
        for k, v in band_map.items():
            new_bands[v] = band_meta[k]
        return new_bands

    if meta["product_type"] == "nbar":
        meta["image"]["bands"] = _replace_bands(meta["image"]["bands"])
    elif meta["product_type"] == "pqa":
        for k, v in meta["lineage"]["source_datasets"].items():
            meta["lineage"]["source_datasets"][k]["image"]["bands"] = _replace_bands(
                v["image"]["bands"]
            )

    metadatafile.write_text(yaml.dump(meta))


def test_end_to_end(clirunner, index, testdata_dir):
    """
    Index two datasets:
        1) Original LS5 NBAR geotif data
        2) zarrified and reprojected (to albers) LS5 NBAR dataset

    This test is the same as ODC test_end_to_end except ingestion to albers is replaced
    with zarrify (incl. reproject) and index.
    """

    lbg_nbar = testdata_dir / 'lbg' / LBG_NBAR
    lbg_pq = testdata_dir / 'lbg' / LBG_PQ

    # Add the LS5 Dataset Types
    clirunner(['-v', 'product', 'add', str(LS5_DATASET_TYPES)])

    # Index the Datasets
    #  - do test run first to increase test coverage
    clirunner(['-v', 'dataset', 'add', '--dry-run', str(lbg_nbar), str(lbg_pq)])

    #  - do actual indexing
    clirunner(['-v', 'dataset', 'add', str(lbg_nbar), str(lbg_pq)])

    #  - this will be no-op but with ignore lineage
    clirunner(
        ['-v', 'dataset', 'add', '--confirm-ignore-lineage', str(lbg_nbar), str(lbg_pq)]
    )

    # Test no-op update
    for policy in ['archive', 'forget', 'keep']:
        clirunner(
            [
                '-v',
                'dataset',
                'update',
                '--dry-run',
                '--location-policy',
                policy,
                str(lbg_nbar),
                str(lbg_pq),
            ]
        )

        # Test no changes needed update
        clirunner(
            [
                '-v',
                'dataset',
                'update',
                '--location-policy',
                policy,
                str(lbg_nbar),
                str(lbg_pq),
            ]
        )

    # TODO: test location update
    # 1. Make a copy of a file
    # 2. Call dataset update with archive/forget
    # 3. Check location

    # Zarrify geotiffs to albers projection, prepare and index
    runner = CliRunner()
    zarrify_args = "--chunk x:500 --chunk y:500 --progress".split()
    zarrify_args.extend("--crs EPSG:3577 --resolution 25 25".split())
    zarr_dir = testdata_dir / "zarrs"
    zarrify_args.extend(["--outpath", str(zarr_dir)])
    gtif_dir = testdata_dir / 'lbg'
    zarrify_args.append(str(gtif_dir))
    res_zarrify = runner.invoke(zarrify, zarrify_args)
    assert res_zarrify.exit_code == 0, res_zarrify.stdout
    zarr_dataset_dir = zarr_dir / "lbg"

    # prepare metadata for zarr
    res_prep = runner.invoke(prepare_zarr_ls5, [str(zarr_dataset_dir / LBG_NBAR)])
    assert res_prep.exit_code == 0, res_prep.stdout
    for metafile in zarr_dataset_dir.glob("**/agdc-metadata.yaml"):
        replace_band_names(metafile)

    # Add the zarr LS5 products
    clirunner(["-v", "product", "add", str(LS5_DATASET_TYPES_ZARR)])
    for ds in (LBG_NBAR, LBG_PQ):
        ds_dir = zarr_dataset_dir / ds
        clirunner(["-v", "dataset", "add", str(ds_dir)])

    dc = Datacube(index=index)
    assert isinstance(str(dc), str)
    assert isinstance(repr(dc), str)

    with pytest.raises(ValueError):
        dc.find_datasets(time='2019')  # no product supplied, raises exception

    product = 'ls5_nbar_scene_zarr'
    check_open_with_dc(index, product=product)
    check_open_with_grid_workflow(index, product=product)
    check_load_via_dss(index, product=product)


def check_open_with_dc(index, product):
    dc = Datacube(index=index)

    data_array = dc.load(product=product, measurements=['blue']).to_array(dim='variable')
    assert data_array.shape
    assert (data_array != -999).any()

    data_array = dc.load(
        product=product, measurements=['blue'], time='1992-03-23T23:14:25.500000'
    )
    assert data_array['blue'].shape[0] == 1
    assert (data_array.blue != -999).any()

    data_array = dc.load(
        product=product, measurements=['blue'], latitude=-35.3, longitude=149.1
    )
    assert data_array['blue'].shape[1:] == (1, 1)
    assert (data_array.blue != -999).any()

    data_array = dc.load(
        product=product, latitude=(-35, -36), longitude=(149, 150)
    ).to_array(dim='variable')

    assert data_array.ndim == 4
    assert 'variable' in data_array.dims
    assert (data_array != -999).any()

    with rasterio.Env():
        lazy_data_array = dc.load(
            product=product,
            latitude=(-35, -36),
            longitude=(149, 150),
            dask_chunks={'time': 1, 'x': 1000, 'y': 1000},
        ).to_array(dim='variable')
        assert lazy_data_array.data.dask
        assert lazy_data_array.ndim == data_array.ndim
        assert 'variable' in lazy_data_array.dims
        assert lazy_data_array[1, :2, 950:1050, 950:1050].equals(
            data_array[1, :2, 950:1050, 950:1050]
        )

    dataset = dc.load(product=product, measurements=['blue'], fuse_func=custom_dumb_fuser)
    assert dataset['blue'].size

    dataset = dc.load(product=product, latitude=(-35.2, -35.3), longitude=(149.1, 149.2))
    assert dataset['blue'].size

    with rasterio.Env():
        lazy_dataset = dc.load(
            product=product,
            latitude=(-35.2, -35.3),
            longitude=(149.1, 149.2),
            dask_chunks={'time': 1},
        )
        assert lazy_dataset['blue'].data.dask
        assert lazy_dataset.blue[:2, :100, :100].equals(dataset.blue[:2, :100, :100])
        assert lazy_dataset.isel(
            time=slice(0, 2), x=slice(950, 1050), y=slice(950, 1050)
        ).equals(dataset.isel(time=slice(0, 2), x=slice(950, 1050), y=slice(950, 1050)))

        # again but with larger time chunks
        lazy_dataset = dc.load(
            product=product,
            latitude=(-35.2, -35.3),
            longitude=(149.1, 149.2),
            dask_chunks={'time': 2},
        )
        assert lazy_dataset['blue'].data.dask
        assert lazy_dataset.blue[:2, :100, :100].equals(dataset.blue[:2, :100, :100])
        assert lazy_dataset.isel(
            time=slice(0, 2), x=slice(950, 1050), y=slice(950, 1050)
        ).equals(dataset.isel(time=slice(0, 2), x=slice(950, 1050), y=slice(950, 1050)))

    dataset_like = dc.load(product=product, measurements=['blue'], like=dataset)
    assert (dataset.blue == dataset_like.blue).all()

    solar_day_dataset = dc.load(
        product=product,
        latitude=(-35, -36),
        longitude=(149, 150),
        measurements=['blue'],
        group_by='solar_day',
    )
    assert 0 < solar_day_dataset.time.size <= dataset.time.size

    dataset = dc.load(
        product=product, latitude=(-35.2, -35.3), longitude=(149.1, 149.2), align=(5, 20)
    )
    assert dataset.geobox.affine.f % abs(dataset.geobox.affine.e) == 5
    assert dataset.geobox.affine.c % abs(dataset.geobox.affine.a) == 20
    dataset_like = dc.load(product=product, measurements=['blue'], like=dataset)
    assert (dataset.blue == dataset_like.blue).all()

    products_df = dc.list_products()
    assert len(products_df)
    assert len(products_df[products_df['name'].isin([product])])
    # assert len(products_df[products_df['name'].isin(['ls5_pq_albers'])])

    assert len(dc.list_measurements())
    assert len(dc.list_measurements(with_pandas=False))
    assert len(dc.list_products(with_pandas=False))

    resamp = ['nearest', 'cubic', 'bilinear', 'cubic_spline', 'lanczos', 'average']
    results = {}

    def calc_max_change(da):
        midline = int(da.shape[0] * 0.5)
        a = int(abs(da[midline, :-1].data - da[midline, 1:].data).max())

        centerline = int(da.shape[1] * 0.5)
        b = int(abs(da[:-1, centerline].data - da[1:, centerline].data).max())
        return a + b

    for resamp_meth in resamp:
        dataset = dc.load(
            product=product,
            measurements=['blue'],
            latitude=(-35.28, -35.285),
            longitude=(149.15, 149.155),
            output_crs='EPSG:4326',
            resolution=(-0.0000125, 0.0000125),
            resampling=resamp_meth,
        )
        results[resamp_meth] = calc_max_change(dataset.blue.isel(time=0))

    assert results['cubic_spline'] < results['nearest']
    assert results['lanczos'] < results['average']

    # check empty result
    dataset = dc.load(
        product=product,
        time=('1918', '1919'),
        measurements=['blue'],
        latitude=(-35.28, -35.285),
        longitude=(149.15, 149.155),
        output_crs='EPSG:4326',
        resolution=(-0.0000125, 0.0000125),
    )
    assert len(dataset.data_vars) == 0


def check_open_with_grid_workflow(index, product):
    dt = index.products.get_by_name(product)

    from datacube.api.grid_workflow import GridWorkflow

    gw = GridWorkflow(index, dt.grid_spec)

    cells = gw.list_cells(product=product, cell_index=LBG_CELL)
    assert LBG_CELL in cells

    cells = gw.list_cells(product=product)
    assert LBG_CELL in cells

    tile = cells[LBG_CELL]
    assert 'x' in tile.dims
    assert 'y' in tile.dims
    assert 'time' in tile.dims
    assert tile.shape[1] == 4000
    assert tile.shape[2] == 4000
    assert tile[:1, :100, :100].shape == (1, 100, 100)
    dataset_cell = gw.load(tile, measurements=['blue'])
    assert dataset_cell['blue'].shape == tile.shape

    for timestamp, tile_slice in tile.split('time'):
        assert tile_slice.shape == (1, 4000, 4000)

    dataset_cell = gw.load(tile)
    assert all(
        m in dataset_cell for m in ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']
    )

    ts = numpy.datetime64('1992-03-23T23:14:25.500000000')
    tile_key = LBG_CELL + (ts,)
    tiles = gw.list_tiles(product=product)
    assert tiles
    assert tile_key in tiles

    tile = tiles[tile_key]
    dataset_cell = gw.load(tile, measurements=['blue'])
    assert dataset_cell['blue'].size

    dataset_cell = gw.load(tile)
    assert all(
        m in dataset_cell for m in ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']
    )


def check_load_via_dss(index, product):
    dc = Datacube(index=index)

    dss = dc.find_datasets(product=product)
    assert len(dss) > 0

    xx1 = dc.load(product=product, measurements=['blue'])
    xx2 = dc.load(datasets=dss, measurements=['blue'])
    assert xx1.blue.shape
    assert (xx1.blue != -999).any()
    assert (xx1.blue == xx2.blue).all()

    xx2 = dc.load(datasets=iter(dss), measurements=['blue'])
    assert xx1.blue.shape
    assert (xx1.blue != -999).any()
    assert (xx1.blue == xx2.blue).all()

    with pytest.raises(ValueError):
        dc.load(measurements=['blue'])
