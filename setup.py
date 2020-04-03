#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages
import runpy
__version__ = runpy.run_path('inferno/version.py')['__version__']


with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    # TODO: put package requirements here
    "pip>=8.1.2",
    "torch>=0.1.12",
    "dill",
    "pyyaml",
    "scipy>=0.13.0",
    "h5py",
    "numpy>=1.8",
    "scikit-image",
    "torchvision",
    "tqdm"
]


setup_requirements = [
    'pytest-runner'
]

test_requirements = [
    'pytest','unittest'
]

dependency_links  = [
    'http://download.pytorch.org/whl/cu75/torch-0.2.0.post1-cp35-cp35m-manylinux1_x86_64.whl#egg=torch-0.2.0'
]

setup(
    name='inferno-pytorch',
    version=__version__,
    description="Inferno is a little library providing utilities and convenience functions/classes around PyTorch.",
    long_description=readme + '\n\n' + history,
    author="Nasim Rahaman",
    author_email='nasim.rahaman@iwr.uni-heidelberg.de',
    url='https://github.com/nasimrahaman/inferno',
    packages=find_packages(where='.',exclude=["*.tests", "*.tests.*", "tests.*", "tests","__pycache__","*.pyc"]),
    dependency_links=dependency_links,
    include_package_data=True,
    install_requires=requirements,
    license="Apache Software License 2.0",
    zip_safe=False,
    keywords='inferno pytorch torch deep learning cnn deep-pyromania',
    classifiers=[
        # How mature is this project? Common values are\
        #   2 - Pre-Alpha',
        #   3 - Alpha,
        #   4 - Beta,
        #   5 - Production/Stable
        'Development Status :: 2 - Pre-Alpha',
        # Indicate who your project is intended for
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6'
    ],
    test_suite='test',
    tests_require=test_requirements,
    setup_requires=setup_requirements,
)
