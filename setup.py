from __future__ import unicode_literals

import os
import sys

from setuptools import find_packages, setup

try:
    import pypandoc
    long_description = pypandoc.convert_file('README.md', 'rst')
except (OSError, IOError, ImportError):
    long_description = open('README.md').read()


def setup_package():
    src_path = os.path.dirname(os.path.abspath(sys.argv[0]))
    old_path = os.getcwd()
    os.chdir(src_path)
    sys.path.insert(0, src_path)

    needs_pytest = {'pytest', 'test', 'ptr'}.intersection(sys.argv)
    pytest_runner = ['pytest-runner>=2.9'] if needs_pytest else []

    setup_requires = ['ncephes>=1.0.21'] + pytest_runner
    install_requires = [
        'scipy>=0.18', 'ncephes>=1.0.40', 'numpy>=1.10', 'numpy-sugar>=1.0.47',
        'scipy-sugar>=1.0.2', 'optimix>=1.2.13', 'cachetools>=2.0', 'tqdm>=4',
        'liknorm-py>=1.0.6'
    ]
    tests_require = ['pytest', 'pytest-pep8']

    metadata = dict(
        name='glimix-core',
        version='1.2.20',
        maintainer="Limix Developers",
        maintainer_email="horta@ebi.ac.uk",
        license="MIT",
        description="Fast inference for Generalized Linear Mixed Models.",
        long_description=long_description,
        url='https://github.com/limix/glimix-core',
        packages=find_packages(),
        zip_safe=False,
        install_requires=install_requires,
        setup_requires=setup_requires,
        tests_require=tests_require,
        include_package_data=True,
        classifiers=[
            "Development Status :: 5 - Production/Stable",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
        ])

    try:
        setup(**metadata)
    finally:
        del sys.path[0]
        os.chdir(old_path)


if __name__ == '__main__':
    setup_package()
