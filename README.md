# glimix-core

[![Travis](https://img.shields.io/travis/com/limix/glimix-core.svg?style=flat-square&label=linux%20%2F%20macos%20build)](https://travis-ci.com/limix/glimix-core) [![AppVeyor](https://img.shields.io/appveyor/ci/Horta/glimix-core.svg?style=flat-square&label=windows%20build)](https://ci.appveyor.com/project/Horta/glimix-core) [![Documentation](https://img.shields.io/readthedocs/glimix-core/latest.svg?style=flat-square)](https://glimix-core.readthedocs.io/)

Fast inference over mean and covariance parameters for Generalised Linear Mixed
Models.

It implements the mathematical tricks of
[FaST-LMM](https://github.com/MicrosoftGenomics/FaST-LMM) for the special case
of Linear Mixed Models with a linear covariance matrix and provides an
interface to perform inference over millions of covariates in seconds.
The Generalised Linear Mixed Model inference is implemented via Expectation
Propagation and also makes use of several mathematical tricks to handle large
data sets with thousands of samples and millions of covariates.

## Install

There are two main ways of installing it.
Via [pip](https://pypi.python.org/pypi/pip):

```bash
pip install glimix-core
```

Or via [conda](http://conda.pydata.org/docs/index.html):

```bash
conda install -c conda-forge glimix-core
```

## Running the tests

After installation, you can test it

```bash
python -c "import glimix_core; glimix_core.test()"
```

as long as you have [pytest](https://docs.pytest.org/en/latest/).

## Usage

Here it is a very simple example to get you started:

```python
>>> from numpy import array, ones
>>> from numpy_sugar.linalg import economic_qs_linear
>>> from glimix_core.lmm import LMM
>>>
>>> X = array([[1, 2], [3, -1], [1.1, 0.5], [0.5, -0.4]], float)
>>> QS = economic_qs_linear(X)
>>> X = ones((4, 1))
>>> y = array([-1, 2, 0.3, 0.5])
>>> lmm = LMM(y, X, QS)
>>> lmm.fit(verbose=False)
>>> lmm.lml()
-2.2726234086180557
```

We  also provide an extensive [documentation](http://glimix-core.readthedocs.org/) about the library.

## Authors

* [Danilo Horta](https://github.com/horta)

## License

This project is licensed under the [MIT License](https://raw.githubusercontent.com/limix/glimix-core/master/LICENSE.md).
