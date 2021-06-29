# Zarr prepare scripts

General steps for editing an existing prepare script to work with a  zarr dataset:
1. Change where the file format is set from `"GeoTiff"` to `"zarr"`.
2. Update the extraction/setting of image properties: `crs`, `transform`, `shape`, `nodata`. These properties should be read from zarr files using `ZarrIO` instead of from geotiffs (using e.g. `rasterio`).
3. Change how measurement paths are set. A measurement referencing a zarr dataset must also include the `"layer"` property which references the zarr variable name. In the case of a geotiff converted with [zarrify](zarrify.md) the zarr array will be named `"array"` (this is the convention used by `zarrify`). E.g. for the LS8 EO3 example below, the metadata will change from:

        measurements:
            blue:
                path: LC08_L1TP_091084_20190205_20190221_01_T1_sr_band2.tif
                ...

    to:

        measurements:
            blue:
                path: LC08_L1TP_091084_20190205_20190221_01_T1_sr_band2.zarr
                layer: array
            ...

4. (Optional) Change the UUID generation if the original/zarr datasets are to coexist.


## LS5 dataset using (deprecated) EO metadata

A zarr prepare script for LS5 data (including `datacube-core`/`datacube-zarr` LS5 test dataset: `tests/data/lbg`) is provided here: [prepare_zarr_ls5.py](/examples/prepare_zarr_ls5.py).

This script was created by editing GA's [galsprepare.py](https://github.com/opendatacube/datacube-dataset-config/blob/master/old-prep-scripts/galsprepare.py) script according to the steps above.



## LS8 dataset using new EO3 metadata

A small test dataset for LS8 surface reflectance data and corresponding EO3 metadata file is located in `tests/data/espa/ls8_sr/`.

An EO3 metadata prepare script for the orginal geotiff data is provided here: [eo3prepare_usgs_espa_ls8c1_l2.py](/examples/eo3/eo3prepare_usgs_espa_ls8c1_l2.py). This was created based on GA's [eodatasets3](https://github.com/GeoscienceAustralia/eo-datasets) library.

An edited version for working with zarr datasets is provided here: [eo3prepare_usgs_espa_ls8c1_l2_zarr.py](/examples/eo3/eo3prepare_usgs_espa_ls8c1_l2.py).


The diff below show the changes described in the steps listed above:

```diff
diff --git a/examples/eo3/eo3prepare_usgs_espa_ls8c1_l2.py b/examples/eo3/eo3prepare_usgs_espa_ls8c1_l2_zarr.py
index ad039fe..d61987e 100644
--- a/examples/eo3/eo3prepare_usgs_espa_ls8c1_l2.py
+++ b/examples/eo3/eo3prepare_usgs_espa_ls8c1_l2_zarr.py
@@ -151,6 +154,26 @@ def read_xml(xml: Path) -> dict:
     return d


+def add_measurements(assmebler: EO3DatasetAssembler, name: str, file_path: Path):
+    """
+    Add zarr file measurements to `EO3DatasetAssembler`.
+
+    This replaces the call to EO3DatasetAssembler.note_measurement() which
+    works for geotiffs only at this stage.
+    """
+    ds = ZarrIO().open_dataset(uri=file_path.as_uri())
+    da = ds["array"]
+    transform = Affine(*da.transform)
+    crs = CRS.from_string(da.crs)
+    grid = GridSpec(da.shape, transform, crs)
+    path = str(file_path.relative_to(assmebler._metadata_path.parent))
+    img = da.values
+    nodata = da.nodatavals[0]
+    assmebler._measurements.record_image(
+        name=name, grid=grid, path=path, img=img, layer="array", nodata=nodata
+    )
+
+
 # 1. Sanity check source metadata
 # 2. Populate EO3DatasetAssembler class from source metadata
 # 3. Call p.done() to validate and write the dataset YAML document
@@ -203,7 +227,7 @@ def prepare_and_write(  # noqa: C901
     data_format = mtl_doc["product_metadata"]["output_format"]
     if data_format.upper() != "GEOTIFF":
         raise NotImplementedError(f"Only GeoTIFF currently supported: {data_format}")
-    file_format = FileFormat.GeoTIFF
+    file_format = FileFormat.Zarr
     # Get and grid cell size
     projection_params = mtl_doc["projection_parameters"]
     if (
@@ -230,7 +254,7 @@ def prepare_and_write(  # noqa: C901
         )
         # Detministic ID based on USGS's product id
         # (which changes when the scene is reprocessed by them)
-        p.dataset_id = uuid.uuid5(USGS_UUID_NAMESPACE, p.label)
+        p.dataset_id = uuid.uuid5(USGS_UUID_NAMESPACE, p.label + "zarr")
         p.product_uri = f"https://products.easi-eo.solutions/{p.product_name}"

         p.platform = mtl_doc["product_metadata"]["spacecraft_id"]
@@ -261,10 +285,10 @@ def prepare_and_write(  # noqa: C901
         )
         p.dataset_version = f"{usgs_collection_number}.0.{p.processed:%Y%m%d}"

-        measurement_map = p.map_measurements_to_files(r'T\d_(\w+).tif')
+        measurement_map = p.map_measurements_to_files(r'T\d_(\w+).zarr')
         for measurement_name, file_location in measurement_map.items():
             logging.debug(f'Measurement map: {measurement_name} > {file_location}')
-            p.note_measurement(measurement_name, file_location)
+            add_measurements(p, measurement_name, file_location)

         p.add_accessory_file("metadata:landsat_mtl", mtl_path.name)
```
