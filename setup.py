#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    # TODO: put package requirements here
    "pip==8.1.2",
    "torch",
    "dill",
    "pyyaml",
    "scipy>=0.13.0",
    "h5py",
    "numpy>=1.8",
    "scikit-image"
]

setup_requirements = [
    # TODO(nasimrahaman): put setup requirements (distutils extensions, etc.) here

]

test_requirements = [
    # TODO: put package test requirements here
    'unittest'
]

setup(
    name='inferno',
    version='0.1.0',
    description="Inferno is a little library providing utilities and convenience functions/classes around PyTorch.",
    long_description=readme + '\n\n' + history,
    author="Nasim Rahaman",
    author_email='nasim.rahaman@iwr.uni-heidelberg.de',
    url='https://github.com/nasimrahaman/inferno',
    packages=find_packages(include=['inferno']),
    include_package_data=True,
    install_requires=requirements,
    license="Apache Software License 2.0",
    zip_safe=False,
    keywords='inferno',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6'
    ],
    test_suite='tests',
    tests_require=test_requirements,
    setup_requires=setup_requirements,
)
