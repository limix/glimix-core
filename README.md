# glimix-core

[![Travis](https://img.shields.io/travis/limix/glimix-core.svg?style=flat-square&label=linux%20%2F%20macos%20build)](https://travis-ci.org/limix/glimix-core) [![AppVeyor](https://img.shields.io/appveyor/ci/Horta/glimix-core.svg?style=flat-square&label=windows%20build)](https://ci.appveyor.com/project/Horta/glimix-core) [![Documentation](https://img.shields.io/readthedocs/glimix-core.svg?style=flat-square&version=stable)](https://glimix-core.readthedocs.io/)

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

We recommend installing it via
[conda](http://conda.pydata.org/docs/index.html):
```bash
conda install -c conda-forge glimix-core
```

Alternatively, glimix-core can also be installed using
[pip](https://pypi.python.org/pypi/pip):
```bash
pip install glimix-core
```

The second installation option requires from the user to install
[liknorm](https://github.com/limix/liknorm) beforehand.

## Running the tests

After installation, you can test it

```
python -c "import glimix_core; glimix_core.test()"
```

as long as you have [pytest](https://docs.pytest.org/en/latest/).

## Authors

* [Danilo Horta](https://github.com/horta)

## License

This project is licensed under the [MIT License](https://raw.githubusercontent.com/limix/glimix-core/master/LICENSE.md).
