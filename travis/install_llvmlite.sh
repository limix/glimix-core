#!/usr/bin/env bash

if [ "${TRAVIS_PYTHON_VERSION}" == "2.7" ];
then
  pip install enum34
fi

wget https://github.com/numba/llvmlite/archive/v0.15.0.tar.gz
tar xzf v0.15.0.tar.gz

pushd llvmlite-0.15.0
CXX_FLTO_FLAGS= LD_FLTO_FLAGS= python setup.py install
popd
