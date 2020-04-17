Open Data Cube Drivers
======================

Overview
========

This provides additional drivers for the Open Data Cube (ODC) project.

For ODC documentation and repository, please see (`ODC documentation <http://datacube-core.readthedocs.io/en/latest/>`__,
`ODC repository <https://github.com/opendatacube/datacube-core/>`__)

Drivers provided
================

  - Zarr 2D storage driver

Requirements
============

System
~~~~~~

-  ODC 1.8+
-  PostgreSQL 9.5+
-  Python 3.6+

Developer setup
===============

1. Install ODC (see `ODC developer setup <https://github.com/opendatacube/datacube-core#developer-setup/>`__)

1. Clone:

   -  ``git clone https://csiro-easi@dev.azure.com/csiro-easi/easi-hub/_git/datacube-drivers``

2. Activate the conda environment you created when installing ODC

::

   conda activate odc

3. Install new drivers from this repository.

::

   cd datacube-drivers
   python setup.py install


4. Run unit tests + PyLint
   ``./check-code.sh``

5. **(or)** Run all tests, including integration tests.

   ``./check-code.sh integration_tests``

   -  Assumes a password-less Postgres database running on localhost called

   ``agdcintegration``

   -  Otherwise copy ``integration_tests/agdcintegration.conf`` to
      ``~/.datacube_integration.conf`` and edit to customise.


Alternatively one can use ``opendatacube/datacube-tests`` docker image to run
tests. This docker includes database server pre-configured for running
integration tests. Add ``--with-docker`` command line option as a first argument
to ``./check-code.sh`` script.

::

   ./check-code.sh --with-docker integration_tests
