#!/usr/bin/env python
# -*- coding: utf-8; mode: python -*-
"""
setup.py script for the SMPyBandits project (https://github.com/SMPyBandits/SMPyBandits)
References:
- https://packaging.python.org/en/latest/distributing/#setup-py
- https://the-hitchhikers-guide-to-packaging.readthedocs.io/en/latest/creation.html#setup-py-description
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
long_description = "M2Learn: An python library for multi-modal data learning."
README = path.join(here, "README.rst")
if path.exists(README):
    with open(README, encoding="utf-8") as f:
        long_description = f.read()
        # print("Using a long_description of length,", len(long_description), "from file", README)  # DEBUG

version = "0.1.0"
try:
    from SMPyBandits import __version__ as version
except ImportError:
    print("Error: cannot import version from M2Learn.")
# FIXME revert when done uploading the first version to PyPI
# version = "0.0.2.dev2"


setup(name="M2Learn",
    version=version,
    description="M2Learn: An python library for multi-modal data learning.",
    long_description=long_description,
    author="Suwen Lin",
    # author_email="slin4 AT nd DOT edu".replace(" AT ", "@").replace(" DOT ", "."),
    url="https://github.com/sumenlin/M2Learn",
    license="new BSD",
    classifiers=[
        "Development Status :: 4 - Beta",
        # "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: New BSD License",
        "Natural Language :: English",
        # "Operating System :: Unix",
        "Programming Language :: Python :: 2.7",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development",
    ],
    keywords = ["data mining","machine learning","multi-modal data"],
    packages=[
        "src",
        "M2Learn"
        "M2Learn.feature",
        "M2Learn.pipeline",
        "M2Learn.prediction",
        "M2Learn.preprocessing",
    ],
    install_requires=[
        "numpy",
        "pandas >= 0.22.0",
        "scikit-learn == 0.20.0",
        "imbalanced-learn == 0.4.0",
    ],
    # package_data={
    #     'M2Learn': [
    #         'LICENSE',
    #         'README.rst',
    #     ]
    # },
    project_urls={  # Optional
        "Bug Reports": "https://github.com/sumenlin/M2Learn/issues",
        "Source":      "https://github.com/sumenlin/M2Learn",
    },
)

# End of setup.py