#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2014 CNRS (Hervé BREDIN - http://herve.niderb.fr)

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

"""

Features.

"""

from __future__ import unicode_literals

import numpy as np
from segment import Segment, SlidingWindow
from timeline import Timeline


class BaseSegmentFeature(object):

    """
    Base class for any segment/feature iterator.
    """

    def __init__(self, uri=None):
        super(BaseSegmentFeature, self).__init__()
        self.__uri = uri

    def __get_uri(self):
        return self.__uri

    def __set_uri(self, value):
        self.__uri = value
    uri = property(fget=__get_uri, fset=__set_uri)
    """Path to (or any identifier of) described resource"""

    def __iter__(self):
        """Segment/feature vector iterator

        Use expression 'for segment, feature_vector in segment_feature'

        This method must be implemented by subclass.

        """
        raise NotImplementedError('Missing method "__iter__".')


class BasePrecomputedSegmentFeature(BaseSegmentFeature):

    """'Segment iterator'-driven precomputed feature iterator.

    Parameters
    ----------
    data : numpy array
        Feature vectors stored in such a way that data[i] is ith feature
        vector.
    segment_iterator : :class:`SlidingWindow` or :class:`Timeline`
        Segment iterator.
        Its length must correspond to `data` length.
    uri : string, optional
        name of (audio or video) described resource

    """

    def __init__(self, data, segment_iterator, uri=None):
        # make sure data does not contain NaN nor inf
        data = np.asarray_chkfinite(data)

        # make sure segment_iterator is actually one of those
        if not isinstance(segment_iterator, (SlidingWindow, Timeline)):
            raise TypeError("segment_iterator must 'Timeline' or "
                            "'SlidingWindow'.")

        # make sure it iterates the correct number of segments
        try:
            N = len(segment_iterator)
        except Exception:
            # an exception is raised by `len(sliding_window)`
            # in case sliding window has infinite end.
            # this is acceptable, no worry...
            pass
        else:
            n = data.shape[0]
            if n != N:
                raise ValueError(
                    "mismatch between number of segments (%d) "
                    "and number of feature vectors (%d)." % (N, n))

        super(BasePrecomputedSegmentFeature, self).__init__(uri=uri)
        self.__data = data
        self._segment_iterator = segment_iterator

    def __get_data(self):
        return self.__data
    data = property(fget=__get_data)
    """Raw feature data (numpy array)"""

    def __iter__(self):
        """Feature vector iterator

        Use expression 'for segment, feature_vector in periodic_feature'

        """

        # get number of feature vectors
        n = self.__data.shape[0]

        for i, segment in enumerate(self._segment_iterator):

            # make sure we do not iterate too far...
            if i >= n:
                break

            # yield current segment and corresponding feature vector
            yield segment, self.__data[i]

    def _segmentToRange(self, segment):
        """
        Parameters
        ----------
        segment : :class:`pyannote.base.segment.Segment`

        Returns
        -------
        i, n : int

        """
        raise NotImplementedError('Missing method "_segmentToRange".')

    def _rangeToSegment(self, i, n):
        """
        Parameters
        ----------
        i, n : int

        Returns
        -------
        segment : :class:`pyannote.base.segment.Segment`

        """
        raise NotImplementedError('Missing method "_rangeToSegment".')

    def __call__(self, subset, mode='loose'):
        """
        Use expression 'feature(subset)'

        Parameters
        ----------
        subset : :class:`pyannote.base.segment.Segment` or
                 :class:`pyannote.base.timeline.Timeline`

        Returns
        -------
        data : numpy array

        """

        if not isinstance(subset, (Segment, Timeline)):
            raise TypeError('')

        if isinstance(subset, Segment):
            i, n = self._segmentToRange(subset)
            indices = range(i, i + n)

        elif isinstance(subset, Timeline):
            indices = []
            for segment in subset.coverage():
                i, n = self._segmentToRange(segment)
                indices += range(i, i + n)

        return np.take(self.__data, indices, axis=0, out=None, mode='clip')


class PeriodicPrecomputedFeature(BasePrecomputedSegmentFeature):

    """'Sliding window'-driven precomputed feature iterator.

    Parameters
    ----------
    data : numpy array
        Feature vectors stored in such a way that data[i] is
        ith feature vector.
    sliding_window : :class:`SlidingWindow`
        Sliding window. Its length must correspond to `data` length
        (or it can be infinite -- ie. sliding_window.end = None)
    uri : string, optional
        name of (audio or video) described resource

    Examples
    --------
        >>> data = ...
        >>> sliding_window = SlidingWindow( ... )
        >>> feature_iterator = PeriodicPrecomputedFeature(data, sliding_window)
        >>> for segment, feature_vector in feature_iterator:
        ...     pass

    """

    def __init__(self, data, sliding_window, uri=None):

        super(PeriodicPrecomputedFeature, self).__init__(
            data, sliding_window, uri=uri
        )

    def __get_sliding_window(self):
        return self._segment_iterator
    sliding_window = property(fget=__get_sliding_window)

    def _segmentToRange(self, segment):
        """
        Parameters
        ----------
        segment : :class:`pyannote.base.segment.Segment`

        Returns
        -------
        i, n : int

        """
        return self.sliding_window.segmentToRange(segment)

    def _rangeToSegment(self, i, n):
        """
        Parameters
        ----------
        i, n : int

        Returns
        -------
        segment : :class:`pyannote.base.segment.Segment`

        """
        return self.sliding_window.rangeToSegment(i, n)


class TimelinePrecomputedFeature(BasePrecomputedSegmentFeature):

    """Timeline-driven precomputed feature iterator.

    Parameters
    ----------
    data : numpy array
        Feature vectors stored in such a way that data[i] is ith feature
        vector.
    timeline : :class:`Timeline`
        Timeline whose length must correspond to `data` length
    uri : string, optional
        name of (audio or video) described resource

    Examples
    --------
        >>> data = ...
        >>> timeline = Timeline( ... )
        >>> feature_iterator = TimelinePrecomputedFeature(data, timeline)
        >>> for segment, feature_vector in feature_iterator:
        ...     pass


    """

    def __init__(self, data, timeline, uri=None):
        super(TimelinePrecomputedFeature, self).__init__(data, timeline,
                                                         uri=uri)

    def __get_timeline(self):
        return self._segment_iterator
    timeline = property(fget=__get_timeline)

    def _segmentToRange(self, segment):
        timeline = self.timeline.crop(segment, mode='loose')
        if timeline:
            # index of first segment in sub-timeline
            first_segment = timeline[0]
            i = self.timeline.index(first_segment)
            # number of segments in sub-timeline
            n = len(timeline)
        else:
            i = 0
            n = 0

        return i, n

    def _rangeToSegment(self, i, n):
        first_segment = self.timeline[i]
        last_segment = self.timeline[i + n]
        return first_segment | last_segment


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
