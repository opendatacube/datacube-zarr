#! /usr/bin/env python3

import datacube

# LS5 NBAR scene params
output_crs = "EPSG:32655"
resolution = (25, -25)
latitude = (-35.282468, -35.315409)
longitude = (149.0689, 149.156705)

# Load data
dc = datacube.Datacube()
data_tiff = dc.load(
    product='usgs_espa_ls8c1_sr',
    latitude=latitude,
    longitude=longitude,
    output_crs=output_crs,
    resolution=resolution
)
data_zarr = dc.load(
    product='usgs_espa_ls8c1_sr_zarr',
    latitude=latitude,
    longitude=longitude,
    output_crs=output_crs,
    resolution=resolution
)

# compare datasets
assert data_zarr.equals(data_tiff)
print(data_zarr)
