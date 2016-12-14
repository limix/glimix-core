from __future__ import unicode_literals

import os
import sys

from setuptools import find_packages, setup

try:
    import pypandoc
    long_description = pypandoc.convert('README.md', 'rst')
except(IOError, ImportError):
    long_description = open('README.md').read()

def setup_package():
    src_path = os.path.dirname(os.path.abspath(sys.argv[0]))
    old_path = os.getcwd()
    os.chdir(src_path)
    sys.path.insert(0, src_path)

    needs_pytest = {'pytest', 'test', 'ptr'}.intersection(sys.argv)
    pytest_runner = ['pytest-runner>=2.9'] if needs_pytest else []

    setup_requires = ['cffi>=1.7', 'six'] + pytest_runner
    install_requires = [
        'pytest>=2.9', 'scipy>=0.17', 'numpy>=1.10',
        'numpy-sugar', 'optimix>=1.1.1', 'cachetools>=2.0'
    ]
    tests_require = ['pytest']

    metadata = dict(
        name='limix-inference',
        version='1.0.2dev0',
        maintainer="Limix Developers",
        maintainer_email="horta@ebi.ac.uk",
        license="MIT",
        url='https://github.com/glimix/limix-inference',
        packages=find_packages(),
        zip_safe=False,
        install_requires=install_requires,
        setup_requires=setup_requires,
        tests_require=tests_require,
        include_package_data=True)

    try:
        setup(**metadata)
    finally:
        del sys.path[0]
        os.chdir(old_path)


if __name__ == '__main__':
    setup_package()
