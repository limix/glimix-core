Installation
------------

For those new to the scientific Python community, we strongly recommend first
getting to know about `Anaconda <https://www.continuum.io/downloads>`_
platform.
In particular, we will refer to its package manager
`Conda <http://conda.pydata.org/docs/intro.html>`_ in this documentation.

Limix-inference can be installed via

.. code-block:: console

    pip install limix-inference

The above command should install the latest Lim version. If that doesn't
happen, try instead

.. code-block:: console

    pip install limix-inference --no-cache-dir

to prevent the use of a cached version in your system. And if you already have
Lim previously installed, you can upgrade it via

.. code-block:: console

    pip install limix-inference --upgrade

In any case, make sure you have the latest version

.. code-block:: console

    python -c "import limix_inference; print('Limix-inference ' + limix_inference.__version__)"

.. program-output:: python -c "import limix_inference; print('Limix-inference ' + limix_inference.__version__)"

and that it is actually working

.. code-block:: console

    python -c "import limix_inference; limix_inference.test()"

.. program-output:: python -c "import limix_inference; limix_inference.test()"
