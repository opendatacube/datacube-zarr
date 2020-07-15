from setuptools import find_packages, setup

tests_require = [
    'black',
    'flask',
    'flake8-isort',
    'hypothesis',
    'isort<5.0.0',
    'mypy',
    'mock',
    'moto',
    'pytest',
    'pytest-cov',
    # --- integration tests ---
    'eodatasets3 @ git+https://github.com/GeoscienceAustralia/eo-datasets.git@eodatasets3#egg=eodatasets3',
]


setup(
    name='datacube_zarr',
    author="CSIRO's' Data61",
    maintainer="CSIRO's' Data61",
    license='Apache License 2.0',
    description="Zarr plug-in driver for datacube",
    long_description=open('readme.md').read(),
    long_description_content_type='text/markdown',
    python_requires='>=3.6.0',
    packages=find_packages(
        exclude=('tests', 'tests.*', 'integration_tests', 'integration_tests.*')
    ),
    use_scm_version={
        'write_to': 'datacube_zarr/_version.py',
        'fallback_version': '0.0.0+no.scm',
    },
    setup_requires=['setuptools_scm'],
    install_requires=[
        'boto3>=1.9.0',
        'click>=5.0',
        'datacube>1.8.0',
        'numcodecs>=0.6.2',
        'rasterio>=1.0.4',
        's3path>=0.1.93',
        's3fs>=0.2.0',
        'xarray>=0.14.1',
        'zarr>=2.3.2',
    ],
    extras_require={'test': tests_require},
    tests_require=tests_require,
    entry_points={
        'console_scripts': ['zarrify = datacube_zarr.tools.zarrify:main'],
        'datacube.plugins.io.read': ['zarr = datacube_zarr.driver:reader_driver_init',],
        'datacube.plugins.io.write': ['zarr = datacube_zarr.driver:writer_driver_init',],
    },
)
