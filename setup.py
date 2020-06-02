from setuptools import find_packages, setup

tests_require = [
    'hypothesis',
    'isort>=4.3.21',
    'mypy',
    'mock',
    'moto',
    'pycodestyle',
    'pylint',
    'pytest',
    'pytest-cov',
]

setup(
    name='datacube_zarr',
    version="1.0",
    author="CSIRO's' Data61",
    maintainer="CSIRO's' Data61",
    license='Apache License 2.0',
    description="Zarr plug-in driver for datacube",
    long_description=open('readme.md').read(),
    long_description_content_type='text/markdown',

    packages=find_packages(
        exclude=(
            'tests', 'tests.*',
            'integration_tests', 'integration_tests.*'
        )
    ),
    install_requires=[
        'zarr',
        'xarray',
        'boto3',
        's3fs',
        'numcodecs',
    ],

    extras_require={
        'test': tests_require,
        'index': ['s3path']
    },
    tests_require=tests_require,

    entry_points={
        'datacube.plugins.io.read': [
            'zarr = zarr_io.driver:reader_driver_init',
        ],
        'datacube.plugins.io.write': [
            'zarr_file = zarr_io.driver:file_writer_driver_init',
            'zarr_s3 = zarr_io.driver:s3_writer_driver_init',
        ]
    }
)
