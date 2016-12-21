#!/usr/bin/env bash

if [ "${TRAVIS_PYTHON_VERSION}" == "2.7" ];
then
  pip install enum34
fi

git clone https://github.com/numba/llvmlite.git --depth 1

pushd llvmlite
CXX_FLTO_FLAGS= LD_FLTO_FLAGS= python setup.py install
popd
