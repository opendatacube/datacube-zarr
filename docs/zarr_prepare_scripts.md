# Zarr prepare scripts

General steps for editing an existing prepare script to work with a  zarr dataset:
1. Change where the file format is set from `"GeoTiff"` to `"zarr"`.
2. Update the extraction/setting of image properties (`crs`, `transform`, `shape`, `nodata`). These properteis should be read from zarr files using `ZarrIO` instead of from geotiffs (using e.g. `rasterio`).
3. Change how measurement paths are set. A measurement referencing a zarr dataset must also include the `"layer"` property which references the zarr variable name. In the typical case of a single-banded geotiff converted with `zarrify` this will be `"band1"`. E.g. for the LS8 EO3 example below, the metadata will change from:

        measurements:
            blue:
                path: LC08_L1TP_091084_20190205_20190221_01_T1_sr_band2.tif
                ...

    to:

        measurements:
            blue:
                path: LC08_L1TP_091084_20190205_20190221_01_T1_sr_band2.zarr
                layer: band1
            ...

4. (Optional) Change the UUID generation if the original/zarr datasets are to coexist.


## LS5 dataset using (deprecated) EO metadata

A zarr prepare script for LS5 data (including `datacube-core`/`datacube-zarr` LS5 test dataset: `tests/data/lbg`) is provided here: [prepare_zarr_ls5.py](/examples/prepare_zarr_ls5.py).

This script was created by editing GA's [galsprepare.py](https://github.com/opendatacube/datacube-dataset-config/blob/master/old-prep-scripts/galsprepare.py) script according to the steps above.



## LS8 dataset using new EO3 metadata

A small test dataset for LS8 surface reflectance data and corresponding EO3 metadata file is located in `tests/data/espa/ls8_sr/`.

An EO3 metadata prepare script for the orginal geotiff data is provided here: [eo3prepare_usgs_espa_ls8c1_l2.py](/examples/eo3_gtif/eo3prepare_usgs_espa_ls8c1_l2.py). This was created based on GA's [eodatasets3](https://github.com/GeoscienceAustralia/eo-datasets) library.

An edited version for working with zarr datasets is provided here: [eo3prepare_usgs_espa_ls8c1_l2_zarr.py](/examples/eo3_zarr/eo3prepare_usgs_espa_ls8c1_l2.py).


The diffs below show the changes described in the steps listed above:

```diff
diff --git a/eo3_gtif/eo3prepare_usgs_espa_ls8c1_l2.py b/eo3_zarr/eo3prepare_usgs_espa_ls8c1_l2_zarr.py
index 32cf8ca..a3ed689 100644
--- a/eo3_gtif/eo3prepare_usgs_espa_ls8c1_l2.py
+++ b/eo3_zarr/eo3prepare_usgs_espa_ls8c1_l2_zarr.py
@@ -15,11 +15,14 @@ from pathlib import Path
 from typing import Dict, Iterable, List, Optional, Union

 import click
+from affine import Affine
 from bs4 import BeautifulSoup
-from eodatasets3.model import FileFormat
+from eodatasets3.images import GridSpec
 from eodatasets3.ui import PathPath
+from rasterio.crs import CRS

-from examples.eo3_gtif.eo3_assemble import EO3DatasetAssembler
+from examples.eo3_zarr.eo3_assemble import EO3DatasetAssembler
+from zarr_io import ZarrIO

 """
 label = Optional. Use as a human-readable version of the dataset ID (unique)
@@ -151,6 +154,24 @@ def read_xml(xml: Path) -> dict:
     return d


+def add_measurements(assmebler: EO3DatasetAssembler, name: str, file_path: Path):
+    """
+    Add zarr file measurements to `EO3DatasetAssembler`.
+
+    This replaces the call to EO3DatasetAssembler.note_measurement() which
+    works for geotiffs only at this stage.
+    """
+    ds = ZarrIO().open_dataset(uri=file_path.as_uri())
+    da = ds["band1"]
+    transform = Affine(*ds.transform)
+    crs = CRS.from_proj4(ds.crs)
+    grid = GridSpec(da.shape, transform, crs)
+    path = str(file_path.relative_to(assmebler._metadata_path.parent))
+    img = da.values
+    nodata = ds.nodatavals[0]
+    assmebler._measurements.record_image(name, grid, path, img, nodata)
+
+
 # 1. Sanity check source metadata
 # 2. Populate EO3DatasetAssembler class from source metadata
 # 3. Call p.done() to validate and write the dataset YAML document
@@ -188,7 +209,7 @@ def prepare_and_write(  # noqa: C901
     data_format = mtl_doc["product_metadata"]["output_format"]
     if data_format.upper() != "GEOTIFF":
         raise NotImplementedError(f"Only GeoTIFF currently supported: {data_format}")
-    file_format = FileFormat.GeoTIFF
+    file_format = "Zarr"
     # Get and grid cell size
     projection_params = mtl_doc["projection_parameters"]
     if (
@@ -213,7 +234,8 @@ def prepare_and_write(  # noqa: C901
         # Detministic ID based on USGS's product id
         # (which changes when the scene is reprocessed by them)
         p.dataset_id = uuid.uuid5(
-            USGS_UUID_NAMESPACE, mtl_doc["metadata_file_info"]["landsat_product_id"]
+            USGS_UUID_NAMESPACE,
+            mtl_doc["metadata_file_info"]["landsat_product_id"] + "zarr",
         )
         p.product_uri = f"https://easi-eo.solutions/product/{p.product_name}"
         p.label = f"{p.product_name}-{mtl_doc['metadata_file_info']['landsat_scene_id']}"
@@ -246,12 +268,10 @@ def prepare_and_write(  # noqa: C901
         )
         p.dataset_version = f"{usgs_collection_number}.0.{p.processed:%Y%m%d}"

-        measurement_map = p.map_measurements_to_files(r'T\d_(\w+).tif')
+        measurement_map = p.map_measurements_to_files(r'T\d_(\w+).zarr')
         for measurement_name, file_location in measurement_map.items():
             logging.debug(f'Measurement map: {measurement_name} > {file_location}')
-            p.note_measurement(
-                measurement_name, file_location,
-            )
+            add_measurements(p, measurement_name, file_location)

         p.add_accessory_file("metadata:landsat_mtl", mtl_path.name)

```
```diff
diff --git a/eo3_gtif/eo3_assemble.py b/eo3_zarr/eo3_assemble.py
index d0ac4ed..abd2350 100644
--- a/eo3_gtif/eo3_assemble.py
+++ b/eo3_zarr/eo3_assemble.py
@@ -222,7 +222,6 @@ class EO3DatasetAssembler(EoFields):
         Dict mapping any band IDs (from band_regex) to measurement names.
         Use where the unique band ID does not directly match a measurement name.
         """  # noqa: E501
-
         # Matching is done by interesection of sets, where a single common element
         # indicates a successful match
         measurement2file = {}
@@ -261,7 +260,7 @@ class EO3DatasetAssembler(EoFields):
                 measurement2file[supplementary[c]] = band_ids[c]
                 continue
             raise RuntimeError(
-                f'No unique match for measurements {mtuple} in files: '
+                f'No unique match for measurements {mtuple} in files:'
                 f'{self._dataset_location}'
             )
         return measurement2file
@@ -281,6 +280,11 @@ class EO3DatasetAssembler(EoFields):
         """
         crs, grid_docs, measurement_docs = self._measurements.as_geo_docs()

+        # Add layer to specify zarr variable to load
+        # This should be added to `MeasurementRecord.record_image()`
+        for md in measurement_docs.values():
+            md.layer = "band1"
+
         if measurement_docs and sort_measurements:
             measurement_docs = dict(sorted(measurement_docs.items()))
```