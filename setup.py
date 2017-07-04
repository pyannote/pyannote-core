#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2014-2017 CNRS

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# AUTHORS
# Hervé BREDIN - http://herve.niderb.fr


import versioneer

from setuptools import setup, find_packages

setup(

    # package
    namespace_packages=['pyannote'],
    packages=find_packages(),
    install_requires=[
        'sortedcontainers >= 1.5.7',
        'numpy >= 1.10.4',
        'pandas >= 0.17.1',
        'xarray >= 0.7.0',
        'networkx >= 1.10.0',
        'simplejson >= 3.8.1',
        'six >= 1.10.0',
        'matplotlib >= 2.0.0',
        'unidecode >= 0.04.20',
    ],
    extras_require={
        'notebook': [
            'pygraphviz >= 1.3.1',
            'pydot2 >= 1.0.33',
        ],
    },
    # versioneer
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),

    # PyPI
    name='pyannote.core',
    description=('Advanced data structures for handling '
                 'temporal segments with attached labels.'),
    author='Hervé Bredin',
    author_email='bredin@limsi.fr',
    url='http://pyannote.github.io/',
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering"
    ],
)
