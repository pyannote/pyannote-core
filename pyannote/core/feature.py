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
########
Features
########

See :class:`pyannote.core.SlidingWindowFeature` for the complete reference.
"""

from __future__ import unicode_literals

import numpy as np
from .segment import Segment
from .segment import SlidingWindow
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
        self.__i = -1

    def __len__(self):
        return self.data.shape[0]

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

    def __iter__(self):
        self.__i = -1
        return self

    def __next__(self):
        self.__i += 1
        try:
            return self.sliding_window[self.__i], self.data[self.__i]
        except IndexError as e:
            raise StopIteration()

    def next(self):
        return self.__next__()

    def iterfeatures(self, window=False):
        """Feature vector iterator

        Parameters
        ----------
        window : bool, optional
            When True, yield both feature vector and corresponding window.
            Default is to only yield feature vector

        """
        nSamples = self.data.shape[0]
        for i in range(nSamples):
            if window:
                yield self.data[i], self.sliding_window[i]
            else:
                yield self.data[i]

    def crop(self, focus, mode='loose', fixed=None, return_data=True):
        """Extract frames

        Parameters
        ----------
        focus : Segment or Timeline
        mode : {'loose', 'strict', 'center'}, optional
            In 'strict' mode, only frames fully included in 'focus' support are
            returned. In 'loose' mode, any intersecting frames are returned. In
            'center' mode, first and last frames are chosen to be the ones
            whose centers are the closest to 'focus' start and end times.
            Defaults to 'loose'.
        fixed : float, optional
            Overrides `Segment` 'focus' duration and ensures that the number of
            returned frames is fixed (which might otherwise not be the case
            because of rounding errors).
        return_data : bool, optional
            Return a numpy array (default). For `Segment` 'focus', setting it
            to False will return a `SlidingWindowFeature` instance.

        Returns
        -------
        data : `numpy.ndarray` or `SlidingWindowFeature`
            Frame features.

        See also
        --------
        SlidingWindow.crop

        """

        if (not return_data) and (not isinstance(focus, Segment)):
            msg = ('"focus" must be a "Segment" instance when "return_data"'
                   'is set to False.')
            raise ValueError(msg)

        ranges = self.sliding_window.crop(focus, mode=mode, fixed=fixed,
                                          return_ranges=True)

        # total number of samples in features
        n_samples = self.data.shape[0]

        # 1 for vector features (e.g. MFCC in pyannote.audio)
        # 2 for matrix features (e.g. grey-level frames in pyannote.video)
        # 3 for 3rd order tensor (e.g. RBG frames in pyannote.video)
        n_dimensions = len(self.data.shape) - 1

        # clip ranges
        clipped_ranges, repeat_first, repeat_last = [], 0, 0
        for start, end in ranges:
            # count number of requested samples before first sample
            repeat_first += min(end, 0) - min(start, 0)
            # count number of requested samples after last sample
            repeat_last += max(end, n_samples) - max(start, n_samples)
            # if all requested samples are out of bounds, skip
            if end < 0 or start >= n_samples:
                continue
            # keep track of non-empty clipped ranges
            clipped_ranges += [[max(start, 0), min(end, n_samples)]]

        if clipped_ranges:
            data = np.vstack(
                [self.data[start: end, :] for start, end in clipped_ranges])
        else:
            # if all ranges are out of bounds, just return empty data
            shape = (0, ) + self.data.shape[1:]
            data = np.empty(shape)

        # corner case when 'fixed' duration cropping is requested:
        # correct number of samples even with out-of-bounds indices
        if fixed is not None:
            data = np.vstack([
                # repeat first sample as many times as needed
                np.tile(self.data[0], (repeat_first, ) + (1,) * n_dimensions),
                data,
                # repeat last sample as many times as needed
                np.tile(self.data[n_samples - 1],
                        (repeat_last,) + (1, ) * n_dimensions)])

        # return data
        if return_data:
            return data

        # wrap data in a SlidingWindowFeature and return
        sliding_window = SlidingWindow(
            start=self.sliding_window[ranges[0][0]].start,
            duration=self.sliding_window.duration,
            step=self.sliding_window.step)
        return SlidingWindowFeature(data, sliding_window)

    def _repr_png_(self):
        from .notebook import repr_feature
        return repr_feature(self)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
