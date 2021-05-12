# ZarrIO examples

## Import ZarrIO()
    from datacube_zarr import ZarrIO
    zio = ZarrIO()

## Saving an xarray.Dataset:
    data = xr.DataArray(np.random.randn(1300, 1300))

    # Save to Disk
    uri = 'file:///root/mydata.zarr#dataset1'
    # Save to S3
    uri = 's3://my-bucket/mydata.zarr#dataset1'

    zio = ZarrIO()
    zio.save_dataset(
        uri=uri,
        dataset=data.to_dataset(name='array1'),
        chunks={'dim_0': 1100, 'dim_1': 1100},
    )

## Opening an xarray.Dataset:
    ds = zio.open_dataset(uri=uri)

## Viewing the Zarr tree
    zio.print_tree(uri=uri)

    /
    └── dataset1
        └── array1 (1300, 1300) float64

## Removing Zarr objects
    zio.clean_store(uri=uri)
