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
# Grant JENKS - http://www.grantjenks.com/

from __future__ import unicode_literals

from .segment import Segment
import numpy as np
import collections as co
import sortedcontainers as sc

class SortedDict(sc.SortedDict):
    length = dict.__len__

class SortedSet(co.MutableSet):
    "Sorted set intended only for segments."
    def __init__(self, segments=()):
        "Initialize sorted set from iterable of segments."
        set_segments = set(segments)
        self._set = set_segments
        self._list = sc.SortedList(set_segments)
        marks = (mark for segment in set_segments for mark in segment)
        self._marks = sc.SortedList(marks)

    def __contains__(self, segment):
        return segment in self._set

    def __iter__(self):
        return iter(self._list)

    def __len__(self):
        return len(self._set)

    length = __len__

    def add(self, segment):
        _set = self._set
        if segment in _set:
            return
        _set.add(segment)
        self._list.add(segment)
        _marks = self._marks
        _marks.add(segment.start)
        _marks.add(segment.end)

    def discard(self, segment):
        _set = self._set
        if segment not in _set:
            return
        _set.remove(segment)
        self._list.remove(segment)
        _marks = self._marks
        _marks.remove(segment.start)
        _marks.remove(segment.end)

    def extent(self):
        if self._set:
            _marks = self._marks
            start = _marks[0]
            end = _marks[-1]
            return Segment(start=start, end=end)
        else:
            return Segment(start=np.inf, end=-np.inf)

    def kth(self, index):
        return self._list[index]

    def co_iter(self, other):
        for alpha in self._list:
            temp = Segment(start=alpha.end, end=alpha.end)
            iterable = other._list.irange(maximum=temp)
            for beta in iterable:
                if alpha.intersects(beta):
                    yield alpha, beta

    def overlapping(self, value):
        segment = Segment(start=value, end=value)
        iterable = self._list.irange(maximum=segment)
        result = [segment.overlaps(value) for segment in iterable]
        return result

    union = co.MutableSet.__or__
