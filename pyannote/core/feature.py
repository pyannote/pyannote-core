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

Features.

"""

from __future__ import unicode_literals

import numpy as np
from .segment import Segment
from .timeline import Timeline


class SlidingWindowFeature(object):

    """Periodic feature vectors

    Parameters
    ----------
    data : (nSamples, nFeatures) numpy array

    sliding_window : SlidingWindow


    """

    def __init__(self, data, sliding_window):
        super(SlidingWindowFeature, self).__init__()
        self.sliding_window = sliding_window
        self.data = data

    def getNumber(self):
        """Number of feature vectors"""
        return self.data.shape[0]

    def getDimension(self):
        """Dimension of feature vectors"""
        return self.data.shape[1]

    def getExtent(self):
        return self.sliding_window.rangeToSegment(0, self.getNumber())

    def __getitem__(self, i):
        """Get ith feature vector"""
        return self.data[i]

    def iterfeatures(self, window=False):
        """Feature vector iterator

        Parameters
        ----------
        window : bool, optional
            When True, yield both feature vector and corresponding window.
            Default is to only yield feature vector

        """
        nSamples = self.data.shape[0]
        for i in xrange(nSamples):
            if window:
                yield self.data[i], self.sliding_window[i]
            else:
                yield self.data[i]

    def crop(self, focus):
        """Get set of feature vector for given segment

        Parameters
        ----------
        focus : Segment or Timeline

        Returns
        -------
        data : numpy array
            (nSamples, nFeatures) numpy array
        """

        n = self.getNumber()

        if isinstance(focus, Segment):
            firstFrame, frameNumber = self.sliding_window.segmentToRange(focus)
            indices = range(
                min(n, max(0, firstFrame)),
                min(n, max(0, firstFrame + frameNumber))
            )

        if isinstance(focus, Timeline):
            indices = []
            for segment in focus.coverage():
                firstFrame, frameNumber = self.sliding_window.segmentToRange(
                    segment)
                indices += range(
                    min(n, max(0, firstFrame)),
                    min(n, max(0, firstFrame + frameNumber))
                )

        return np.take(self.data, indices, axis=0, out=None, mode='clip')


if __name__ == "__main__":
    import doctest
    doctest.testmod()
