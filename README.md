# limix-inference

[![PyPIl](https://img.shields.io/pypi/l/limix-inference.svg?style=flat-square)](https://pypi.python.org/pypi/limix-inference/)
[![PyPIv](https://img.shields.io/pypi/v/limix-inference.svg?style=flat-square)](https://pypi.python.org/pypi/limix-inference/)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/limix-inference/badges/version.svg)](https://anaconda.org/conda-forge/limix-inference)
[![Documentation Status](https://readthedocs.org/projects/limix-inference/badge/?style=flat-square&version=latest)](https://limix-inference.readthedocs.io/en/latest/)


Fast inference for Generalized Linear Mixed Models.

## Install

The recommended way of installing it is via
[conda](http://conda.pydata.org/docs/index.html)
```bash
conda install -c conda-forge limix-inference
```

An alternative way would be via [pip](https://pypi.python.org/pypi/pip).
First you need to install [liknorm](http://liknorm.readthedocs.io/en/latest/)
library and then
```bash
pip install limix-inference
```

Refer to [documentation](http://limix-inference.readthedocs.io/en/latest/)
for more information.

## Running the tests

After installation, you can test it
```
python -c "import limix_inference; limix_inference.test()"
```
as long as you have [pytest](http://docs.pytest.org/en/latest/).

## Authors

* **Danilo Horta** - [https://github.com/Horta](https://github.com/Horta)

## License

This project is licensed under the MIT License - see the
[LICENSE](LICENSE) file for details
