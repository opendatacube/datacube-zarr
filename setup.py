from setuptools import find_packages, setup

tests_require = [
    'click',
    'flask',
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

    use_scm_version={
        'write_to': 'zarr_io/_version.py',
        'fallback_version': '0.0.0+no.scm',
    },
    setup_requires=['setuptools_scm'],
    install_requires=[
        'boto3>=1.9',
        'numcodecs>=0.6.2',
        'rasterio>=1.0.4',
        's3path>=0.1.93',
        's3fs>=0.2',
        'xarray>=0.14.1',
        'zarr>=2.3.2',
    ],
    extras_require={
        'test': tests_require,
        'tools': ['click'],
    },
    tests_require=tests_require,

    entry_points={
        'datacube.plugins.io.read': [
            'zarr = zarr_io.driver:reader_driver_init',
        ],
        'datacube.plugins.io.write': [
            'zarr = zarr_io.driver:writer_driver_init',
        ]
    }
)
