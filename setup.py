from setuptools import find_packages, setup

tests_require = [
    'black>=20.8b1',
    'flake8>=3.8.0',
    'flake8-isort>=4.0.0',
    'hypothesis',
    'isort>=5.1.0',
    'mypy',
    'mock',
    'moto[server,s3]>2',
    'pytest',
    'pytest-cov',
    'eodatasets3',
    'GDAL',
    'lxml',
    'beautifulsoup4',
]


setup(
    name='datacube_zarr',
    author="CSIRO's' Data61",
    maintainer="CSIRO's' Data61",
    license='Apache License 2.0',
    description="Zarr plug-in driver for datacube",
    long_description=open('readme.md').read(),
    long_description_content_type='text/markdown',
    python_requires='>=3.8.0',
    packages=find_packages(
        exclude=('tests', 'tests.*', 'integration_tests', 'integration_tests.*')
    ),
    use_scm_version={
        'write_to': 'datacube_zarr/_version.py',
        'fallback_version': '0.0.0+no.scm',
    },
    setup_requires=['setuptools_scm'],
    install_requires=[
        'click>=5.0',
        'datacube>1.8.0',
        'numcodecs>=0.6.2',
        'rasterio>=1.1.5',
        's3path>=0.1.93',
        's3fs[boto3,awscli]>=2021.04.0',
        'xarray>=0.16.2',
        'zarr>=2.3.2',
    ],
    extras_require={'test': tests_require},
    tests_require=tests_require,
    entry_points={
        'console_scripts': [
            'zarrify = datacube_zarr.tools.zarrify:cli',
        ],
        'datacube.plugins.io.read': [
            'zarr = datacube_zarr.driver:reader_driver_init',
        ],
    },
)
