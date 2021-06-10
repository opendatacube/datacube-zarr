import os
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio
from datacube.utils.documents import load_from_yaml

#: Number of bands to place in generated GeoTIFFs
NUM_BANDS = 3


GEOTIFF = {
    'date': datetime(1990, 3, 2),
    'shape': {'x': 432, 'y': 321},
    'pixel_size': {'x': 25.0, 'y': -25.0},
    'crs': 'EPSG:28355',  # 'EPSG:28355'
    'ul': {
        'x': 638000.0,  # Coords must match crs
        'y': 6276000.0,  # Coords must match crs
    },
}


def copytree(p1: Path, p2: Path) -> None:
    """Copytree for local/s3 paths."""
    for o1 in p1.iterdir():
        o2 = p2 / o1.name
        if o1.is_dir():
            copytree(o1, o2)
        else:
            if o2.as_uri().startswith("file") and not o2.parent.exists():
                o2.parent.mkdir(parents=True)
            o2.write_bytes(o1.read_bytes())


def _make_geotiffs(tiffs_dir, day_offset, num_bands=NUM_BANDS):
    """
    Generate custom geotiff files, one per band.

    Create ``num_bands`` TIFF files inside ``tiffs_dir``.

    Return a dictionary mapping band_number to filename, eg::

        {
            0: '/tmp/tiffs/band01_time01.tif',
            1: '/tmp/tiffs/band02_time01.tif'
        }
    """
    tiffs = {}
    width = GEOTIFF['shape']['x']
    height = GEOTIFF['shape']['y']
    metadata = {
        'count': 1,
        'crs': GEOTIFF['crs'],
        'driver': 'GTiff',
        'dtype': 'int16',
        'width': width,
        'height': height,
        'nodata': -999.0,
        'transform': [
            GEOTIFF['pixel_size']['x'],
            0.0,
            GEOTIFF['ul']['x'],
            0.0,
            GEOTIFF['pixel_size']['y'],
            GEOTIFF['ul']['y'],
        ],
    }

    for band in range(num_bands):
        path = str(tiffs_dir.join('band%02d_time%02d.tif' % ((band + 1), day_offset)))
        with rasterio.open(path, 'w', **metadata) as dst:
            # Write data in "corners" (rounded down by 100, for a size of 100x100)
            data = np.zeros((height, width), dtype=np.int16)
            data[:] = (
                np.arange(height * width).reshape((height, width))
                + 10 * band
                + day_offset
            )
            '''
            lr = (100 * int(GEOTIFF['shape']['y'] / 100.0),
                  100 * int(GEOTIFF['shape']['x'] / 100.0))
            data[0:100, 0:100] = 100 + day_offset
            data[lr[0] - 100:lr[0], 0:100] = 200 + day_offset
            data[0:100, lr[1] - 100:lr[1]] = 300 + day_offset
            data[lr[0] - 100:lr[0], lr[1] - 100:lr[1]] = 400 + day_offset
            '''
            dst.write(data, 1)
        tiffs[band] = path
    return tiffs


def load_yaml_file(filename):
    with open(str(filename)) as f:
        return list(load_from_yaml(f, parse_dates=True))
