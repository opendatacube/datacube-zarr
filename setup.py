from setuptools import find_packages, setup

setup(
    name='datacube_zarr',
    version="1.0",
    description="Zarr plug-in driver for datacube",
    author='Data61 - CSIRO',
    packages=find_packages(),

    install_requires=[
        'zarr',
        'xarray',
        'boto3',
        'botocore',
        's3fs',
        'numcodecs',
        'mypy'
    ],

    entry_points={
        'datacube.plugins.io.read': [
            'zarr_file = zarr_io.driver:file_reader_driver_init',
            'zarr_s3 = zarr_io.driver:s3_reader_driver_init',
        ],
        'datacube.plugins.io.write': [
            'zarr_file = zarr_io.driver:file_writer_driver_init',
            'zarr_s3 = zarr_io.driver:s3_writer_driver_init',
        ]
    }
)
