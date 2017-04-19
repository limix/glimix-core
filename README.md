# glimix-core

[![PyPIl](https://img.shields.io/pypi/l/glimix-core.svg?style=flat-square)](https://pypi.python.org/pypi/glimix-core/)
[![PyPIv](https://img.shields.io/pypi/v/glimix-core.svg?style=flat-square)](https://pypi.python.org/pypi/glimix-core/)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/glimix-core/badges/version.svg)](https://anaconda.org/conda-forge/glimix-core)
[![Documentation Status](https://readthedocs.org/projects/glimix-core/badge/?style=flat-square&version=latest)](https://glimix-core.readthedocs.io/en/latest/)


Fast inference for Generalized Linear Mixed Models.

## Install

The recommended way of installing it is via
[conda](http://conda.pydata.org/docs/index.html)
```bash
conda install -c conda-forge glimix-core
```

An alternative way would be via [pip](https://pypi.python.org/pypi/pip).
First you need to install [liknorm](http://liknorm.readthedocs.io/en/latest/)
library and then
```bash
pip install glimix-core
```

Refer to [documentation](http://glimix-core.readthedocs.io/en/latest/)
for more information.

## Running the tests

After installation, you can test it
```
python -c "import glimix_core; glimix_core.test()"
```
as long as you have [pytest](http://docs.pytest.org/en/latest/).

## Authors

* **Danilo Horta** - [https://github.com/Horta](https://github.com/Horta)

## License

This project is licensed under the MIT License - see the
[LICENSE](LICENSE) file for details
