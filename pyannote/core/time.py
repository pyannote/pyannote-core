#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2014 CNRS

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

"""
Provides three objects: T, TStart and TEnd

TStart and TEnd are used to represent start ($) and end (^) time of a medium.

T provides facilities to generate anchored or drifting times.
A time is `anchored` if its temporal position is known precisely (e.g. t=3s).
A time is `drifting` if its exact position is not yet known.

To generate an `anchored` time, simply use:
>>> t = T(3.)

To generate a `drifting` time, simply use:
>>> t = T()

Each subsequent call to T() increments an internal string providing a unique
identifier to each new `drifting` time.
>>> print T()
A
>>> print T()
B

One can reset the internal states if needed.
>>> T.reset()
>>> print T()
A

Also, if you want to refer to a specific `drifting` time, simple use:
>>> t = T('B')
"""

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from __future__ import unicode_literals
import itertools

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class _TAnchored(float):

    def __new__(cls, t):
        return float.__new__(cls, t)

    @property
    def anchored(self):
        return True

    @property
    def drifting(self):
        return False

    def __str__(self):

        if float(self) == float('infinity'):
            return '^'

        if float(self) == -float('infinity'):
            return '$'

        return super(_TAnchored, self).__str__()

    def __lt__(self, other):
        if isinstance(other, _TDrifting):
            return True
        return float(self) < float(other)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

def _t_iter():
    """Label generator

    Usage
    -----
    t = _t_iter()
    next(t) -> 'A'    # start with 1-letter labels
    ...               # from A to Z
    next(t) -> 'Z'
    next(t) -> 'AA'   # then 2-letters labels
    next(t) -> 'AB'   # from AA to ZZ
    ...
    next(t) -> 'ZY'
    next(t) -> 'ZZ'
    next(t) -> 'AAA'  # then 3-letters labels
    ...               # (you get the idea)
    """

    # label length
    r = 1

    # infinite loop
    while True:

        # generate labels with current length
        for c in itertools.product(ALPHABET, repeat=r):
            yield "".join(c)

        # increment label length when all possibilities are exhausted
        r = r + 1


class _TDrifting(str):

    __t_iter = _t_iter()

    def __new__(cls, t=None):
        if t is None:
            t = next(cls.__t_iter)
        return str.__new__(cls, t)

    @classmethod
    def _reset(cls):
        """Reset label generator"""
        cls.__t_iter = _t_iter()

    @property
    def anchored(self):
        return False

    @property
    def drifting(self):
        return True

    def __lt__(self, other):
        if isinstance(other, _TAnchored):
            return False
        return str(self) < str(other)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class _T(object):
    """
    Facility to generate `anchored` or `drifting` times.
    A time is `anchored` if its temporal position is known precisely (e.g. t=3s).
    A time is `drifting` if its exact position is not yet known.

    To generate an `anchored` time, simply use:
    >>> t = T(3.)

    To generate a `drifting` time, simply use:
    >>> t = T()

    Each subsequent call to T() increments an internal string providing a unique
    identifier to each new `drifting` time.
    >>> print T()
    A
    >>> print T()
    B

    One can reset the internal states if needed.
    >>> T.reset()
    >>> print T()
    A

    Also, if you want to refer to a specific `drifting` time, simple use:
    >>> t = T('B')
    """

    def __call__(self, t=None):
        if t is None or isinstance(t, str):
            return _TDrifting(t)
        else:
            return _TAnchored(float(t))

    def reset(self):
        """Reset drifting time string generator"""
        _TDrifting._reset()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

__all__ = ['T', 'TStart', 'TEnd']

T = _T()

TStart = _TAnchored(-float('infinity'))  # == T(-float('infinity'))
TEnd = _TAnchored(float('infinity'))     # == T(+float('infinity'))
