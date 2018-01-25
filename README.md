
glimix-core
===========

.. image:: https://github.com/limix/glimix-core/blob/develop/logo.png

|PyPI-Status| |Conda-Forge-Status| |Conda-Downloads| |Build-Status| |Codacy-Grade| |License-Badge| |Doc-Status|

Fast inference over mean and covariance parameters for Generalised Linear Mixed Models.
It implements the mathematical tricks of FaST-LMM_ for the special case of Linear Mixed Models
with a linear covariance matrix and provides an interface to perform inference over millions of
covariates in seconds. (Refer to FastScanner_ for details.)
The Generalised Linear Mixed Model inference is implemented via Expectation Propagation and
also makes use of several mathematical tricks to handle large data sets with thousands of samples
and millions of covariates. (Refer to GLMMExpFam_ and FastScanner_ for details.)

Install
-------

The recommended way of installing it is via conda_

.. code:: bash

    conda install -c conda-forge glimix-core

An alternative way would be via pip_

.. code:: bash

    pip install glimix-core

Running the tests
-----------------

After installation, you can test it

.. code:: bash

    python -c "import glimix_core; glimix_core.test()"

as long as you have pytest_.

Authors
-------

* `Danilo Horta`_

License
-------

This project is licensed under the MIT License - see the `License file`_ file
for details.

.. |Build-Status| image:: https://travis-ci.org/limix/glimix-core.svg?branch=master
    :target: https://travis-ci.org/limix/glimix-core

.. |Codacy-Grade| image:: https://api.codacy.com/project/badge/Grade/e0227434c8f040888ff92d1a4d67bcc8
    :target: https://www.codacy.com/app/danilo.horta/glimix-core?utm_source=github.com&utm_medium=referral&utm_content=limix/glimix-core&utm_campaign=badger

.. |PyPI-Status| image:: https://img.shields.io/pypi/v/glimix-core.svg
    :target: https://pypi.python.org/pypi/glimix-core

.. |PyPI-Versions| image:: https://img.shields.io/pypi/pyversions/glimix-core.svg
    :target: https://pypi.python.org/pypi/glimix-core

.. |Conda-Forge-Status| image:: https://anaconda.org/conda-forge/glimix-core/badges/version.svg
    :target: https://anaconda.org/conda-forge/glimix-core

.. |Conda-Downloads| image:: https://anaconda.org/conda-forge/glimix-core/badges/downloads.svg?style=flat
    :target: https://anaconda.org/conda-forge/glimix-core

.. |License-Badge| image:: https://img.shields.io/pypi/l/glimix-core.svg
    :target: https://raw.githubusercontent.com/limix/glimix-core/master/LICENSE.txt

.. |Doc-Status| image:: https://readthedocs.org/projects/glimix-core/badge/?style=flat&version=stable
    :target: https://glimix-core.readthedocs.io/

.. _License file: https://raw.githubusercontent.com/limix/glimix-core/master/LICENSE.txt

.. _Danilo Horta: https://github.com/horta

.. _conda: http://conda.pydata.org/docs/index.html

.. _pip: https://pypi.python.org/pypi/pip

.. _pytest: http://docs.pytest.org/en/latest/

.. _FaST-LMM: https://github.com/MicrosoftGenomics/FaST-LMM

.. _FastScanner: http://glimix-core.readthedocs.io/en/stable/lmm.html#glimix_core.lmm.FastScanner

.. _GLMMExpFam: http://glimix-core.readthedocs.io/en/stable/glmm.html#glmmexpfam-class
