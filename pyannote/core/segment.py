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


from __future__ import unicode_literals

from collections import namedtuple
import numpy as np

SEGMENT_PRECISION = 1e-6


class Segment(namedtuple('Segment', ['start', 'end'])):
    """
    Temporal interval defined by its `start` and `end` times.

    Multiple segment operators are available -- including intersection (&),
    inclusion (in), emptiness test, start/end time shifting (+, -, >>, <<).
    They are illustrated in **Examples** section.

    Comparison of two segments is also available (==, !=, <, <=, >, >=).
    Two segments are equal iff they have identical start and end times.
    Segment S is smaller than segment T iff S.start < T.start or if they have
    the same start time and S.end < T.start.

    Parameters
    ----------
    start, end : float
        `start` and `end` times, in seconds.

    Returns
    -------
    segment : Segment
        New segment with `start` and `end` times.

    Examples
    --------
    Create a new temporal interval between 00:13.000 and 00:37.000.

        >>> segment = Segment(start=13., end=37)
        >>> print segment
        [13.000 --> 37.000]

    Inclusion, intersection, union & gap

        >>> s1 = Segment(1, 2)
        >>> s2 = Segment(0, 3)
        >>> if s1 in s2:
        ...    print "Segment %s is included in segment %s." % (s1, s2)
        Segment [1.000 --> 2.000] is included in segment [0.000 --> 3.000].
        >>> s3 = Segment(2, 5)
        >>> print s1 & s3
        ∅
        >>> print s2 & s3
        [2.000 --> 3.000]
        >>> print s2 | s3
        [0.000 --> 5.000]
        >>> print s1 ^ Segment(5, 7)
        [2.000 --> 5.000]

    Test whether segment is empty or not.

        >>> if not Segment(10, 10):
        ...    print "Segment is empty."
        Segment is empty.

    Comparison

        >>> s1 = Segment(1, 3)
        >>> s2 = Segment(1, 3)
        >>> s3 = Segment(2, 6)
        >>> s4 = Segment(1, 2)
        >>> for s in sorted([s1, s2, s3, s4]):
        ...    print s
        [1.000 --> 2.000]
        [1.000 --> 3.000]
        [1.000 --> 3.000]
        [2.000 --> 6.000]

    """

    def __new__(cls, start=0., end=0.):
        # add default values
        return super(Segment, cls).__new__(cls, start, end)

    def __nonzero__(self):
        return self.__bool__()

    def __bool__(self):
        """Use the expression 'if segment'

        Returns
        -------
        valid : bool
            False is segment is empty, True otherwise.

        """
        return bool((self.end - self.start) > SEGMENT_PRECISION)

    def _get_duration(self):
        return self.end - self.start if self else 0.
    duration = property(fget=_get_duration)
    """Get segment duration, in seconds."""

    def _get_middle(self):
        return .5 * (self.start + self.end)
    middle = property(fget=_get_middle)
    """Get segment middle time, in seconds."""

    def __iter__(self):
        """Makes sure tuple(segment) is a tuple of float"""
        yield float(self.start)
        yield float(self.end)

    def copy(self):
        """Duplicate segment."""
        return Segment(start=self.start, end=self.end)

    # ------------------------------------------------------- #
    # Inclusion (in), intersection (&), union (|) and gap (^) #
    # ------------------------------------------------------- #

    def __contains__(self, other):
        """Use the expression 'other in segment'

        Returns
        -------
        contains : bool
            True if other segment is fully included, False otherwise

        """
        return (self.start <= other.start) and (self.end >= other.end)

    def __and__(self, other):
        """Use the expression 'segment & other'

        Returns
        -------
        segment : Segment
            Intersection of the two segments

        """
        start = max(self.start, other.start)
        end = min(self.end, other.end)
        return Segment(start=start, end=end)

    def intersects(self, other):
        """Check whether two segments intersect each other

        Parameters
        ----------
        other : Segment
            Other segment

        Returns
        -------
        intersects : bool
            True if segments intersect, False otherwise
        """
        if not self or not other:
            return False

        return (self.start == other.start) or \
               (self.start < other.start and
                other.start < self.end - SEGMENT_PRECISION) or \
               (self.start > other.start and
                self.start < other.end - SEGMENT_PRECISION)

    def overlaps(self, t):
        return self.start <= t and self.end >= t

    def __or__(self, other):
        """Use the expression 'segment | other'

        Returns
        -------
        segment : Segment
            Shortest segment that contains both segments

        """
        # if segment is empty, union is the other one
        if not self:
            return other
        # if other one is empty, union is self
        if not other:
            return self

        # otherwise, do what's meant to be...
        start = min(self.start, other.start)
        end = max(self.end, other.end)
        return Segment(start=start, end=end)

    def __xor__(self, other):
        """Use the expression 'segment ^ other'

        Returns
        -------
        segment : Segment
            Gap between the two segments

        """
        # if segment is empty, xor is not defined
        if (not self) or (not other):
            raise ValueError('')

        start = min(self.end, other.end)
        end = max(self.start, other.start)
        return Segment(start=start, end=end)

    def __str__(self):
        """Use the expression str(segment)"""
        if self:
            return '[%.3f --> %.3f]' % (self.start, self.end)
        else:
            return '[]'

    def _pretty(self, seconds):
        from datetime import timedelta
        td = timedelta(seconds=seconds)
        days = td.days
        seconds = td.seconds
        microseconds = td.microseconds
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        if abs(days) > 0:
            return '%d:%02d:%02d:%02d.%03d' % (days, hours, minutes,
                                               seconds, microseconds / 1000)
        else:
            return '%02d:%02d:%02d.%03d' % (hours, minutes, seconds,
                                            microseconds / 1000)

    def pretty(self):
        """Human-readable representation of segments"""
        return '[%s --> %s]' % (self._pretty(self.start),
                                self._pretty(self.end))

    def __repr__(self):
        return '<Segment(%g, %g)>' % (self.start, self.end)

    def for_json(self):
        return {'start': self.start, 'end': self.end}

    @classmethod
    def from_json(cls, data):
        return cls(start=data['start'], end=data['end'])

    def _repr_png_(self):
        from .notebook import repr_segment
        return repr_segment(self)


class SlidingWindow(object):
    """Sliding window

    Parameters
    ----------
    duration : float > 0, optional
        Window duration, in seconds. Default is 30 ms.
    step : float > 0, optional
        Step between two consecutive position, in seconds. Default is 10 ms.
    start : float, optional
        First start position of window, in seconds. Default is 0.
    end : float > `start`, optional
        Default is infinity (ie. window keeps sliding forever)

    Examples
    --------

    >>> sw = SlidingWindow(duration, step, start)
    >>> frame_range = (a, b)
    >>> frame_range == sw.toFrameRange(sw.toSegment(*frame_range))
    ... True

    >>> segment = Segment(A, B)
    >>> new_segment = sw.toSegment(*sw.toFrameRange(segment))
    >>> abs(segment) - abs(segment & new_segment) < .5 * sw.step

    """

    def __init__(self, duration=0.030, step=0.010, start=0.000, end=None):
        super(SlidingWindow, self).__init__()

        # duration must be a float > 0
        if duration <= 0:
            raise ValueError("'duration' must be a float > 0.")
        self.__duration = duration

        # step must be a float > 0
        if step <= 0:
            raise ValueError("'step' must be a float > 0.")
        self.__step = step

        # start must be a float.
        self.__start = start

        # if end is not provided, set it to infinity
        if end is None:
            self.__end = np.inf
        else:
            # end must be greater than start
            if end <= start:
                raise ValueError("'end' must be greater than 'start'.")
            self.__end = end

    def __get_start(self):
        return self.__start
    start = property(fget=__get_start)
    """Sliding window start time in seconds."""

    def __get_end(self):
        return self.__end
    end = property(fget=__get_end)
    """Sliding window end time in seconds."""

    def __get_step(self):
        return self.__step
    step = property(fget=__get_step)
    """Sliding window step in seconds."""

    def __get_duration(self):
        return self.__duration
    duration = property(fget=__get_duration)
    """Sliding window duration in seconds."""

    def __closest_frame(self, t):
        """Closest frame to timestamp.

        Parameters
        ----------
        t : float
            Timestamp, in seconds.

        Returns
        -------
        index : int
            Index of frame whose middle is the closest to `timestamp`

        """
        return int(np.rint(
            (t - self.__start - .5 * self.__duration) / self.__step
        ))

    def crop(self, focus, mode='loose', fixed=None):
        """Crop sliding window

        Parameters
        ----------
        focus : `Segment` or `Timeline`
        mode : {'strict', 'loose', 'center', 'fixed'}, optional
            In 'strict' mode, only indices of segments fully included in focus
            coverage are returned. In 'loose' mode, indices of any intersecting
            segment are returned. In 'center' mode, first and last positions are
            chosen to be the positions whose centers are the closest to the
            focus start and end times. Defaults to 'loose'.
        fixed : float, optional
            When provided and mode is 'center', overrides focus duration. This
            might be useful to avoid float rounding errors and to make sure the
            number of positions is deterministic.

        Returns
        -------
        indices : np.array
            Array of unique indices of matching segments
        """

        from .timeline import Timeline

        if isinstance(focus, Segment):

            if mode == 'loose':

                # find smallest integer i such that
                # self.start + i x self.step + self.duration >= focus.start
                i_ = (focus.start - self.duration - self.start) / self.step
                i = int(np.ceil(i_))

                # find largest integer j such that
                # self.start + j x self.step <= focus.end
                j_ = (focus.end - self.start) / self.step
                j = int(np.floor(j_))

                return np.array(range(i, j + 1), dtype=np.int64)

            elif mode == 'strict':

                # find smallest integer i such that
                # self.start + i x self.step >= focus.start
                i_ = (focus.start - self.start) / self.step
                i = int(np.ceil(i_))

                # find largest integer j such that
                # self.start + j x self.step + self.duration <= focus.end
                j_ = (focus.end - self.duration - self.start) / self.step
                j = int(np.floor(j_))

                return np.array(range(i, j + 1), dtype=np.int64)

            elif mode == 'center':

                # find window position whose center is the closest to focus.start
                i = self.__closest_frame(focus.start)

                if fixed is None:
                    # find window position whose center is the closest to focus.end
                    j = self.__closest_frame(focus.end)

                else:
                    # make sure the number of returned position is fixed
                    n_ = np.rint(fixed / self.step)
                    j = i + int(n_)

                return np.array(range(i, j + 1), dtype=np.int64)

            else:
                raise ValueError('mode must be "loose", "strict", or "center"')

        elif isinstance(focus, Timeline):
            return np.unique(np.hstack([
                self.crop(s, mode=mode, fixed=fixed) for s in focus.coverage()]))

        else:
            raise TypeError('focus must be a Segment or a Timeline.')

    def segmentToRange(self, segment):
        """Convert segment to 0-indexed frame range

        Parameters
        ----------
        segment : Segment

        Returns
        -------
        i0 : int
            Index of first frame
        n : int
            Number of frames

        Examples
        --------

            >>> window = SlidingWindow()
            >>> print window.segmentToRange(Segment(10, 15))
            i0, n

        """
        # find closest frame to segment start
        i0 = self.__closest_frame(segment.start)

        # number of steps to cover segment duration
        n = int(segment.duration / self.step) + 1

        return i0, n

    def rangeToSegment(self, i0, n):
        """Convert 0-indexed frame range to segment

        Each frame represents a unique segment of duration 'step', centered on
        the middle of the frame.

        The very first frame (i0 = 0) is the exception. It is extended to the
        sliding window start time.

        Parameters
        ----------
        i0 : int
            Index of first frame
        n : int
            Number of frames

        Returns
        -------
        segment : Segment

        Examples
        --------

            >>> window = SlidingWindow()
            >>> print window.rangeToSegment(3, 2)
            [ --> ]

        """

        # frame start time
        # start = self.start + i0 * self.step
        # frame middle time
        # start += .5 * self.duration
        # subframe start time
        # start -= .5 * self.step
        start = self.__start + (i0 - .5) * self.__step + .5 * self.__duration
        duration = n * self.__step
        end = start + duration

        # extend segment to the beginning of the timeline
        if i0 == 0:
            start = self.start

        return Segment(start, end)

    def samplesToDuration(self, nSamples):
        """Returns duration of samples"""
        return self.rangeToSegment(0, nSamples).duration

    def durationToSamples(self, duration):
        """Returns samples in duration"""
        return self.segmentToRange(Segment(0, duration))[1]

    def __getitem__(self, i):
        """
        Parameters
        ----------
        i : int
            Index of sliding window position

        Returns
        -------
        segment : :class:`Segment`
            Sliding window at ith position

        """

        # window start time at ith position
        start = self.__start + i * self.__step

        # in case segment starts after the end,
        # return an empty segment
        if start >= self.__end:
            return None

        return Segment(start=start, end=start + self.__duration)

    def __iter__(self):
        """Sliding window iterator

        Use expression 'for segment in sliding_window'

        Examples
        --------

            >>> window = SlidingWindow(end=0.1)
            >>> for segment in window:
            ...     print segment
            [0.000 --> 0.030]
            [0.010 --> 0.040]
            [0.020 --> 0.050]
            [0.030 --> 0.060]
            [0.040 --> 0.070]
            [0.050 --> 0.080]
            [0.060 --> 0.090]
            [0.070 --> 0.100]
            [0.080 --> 0.100]
            [0.090 --> 0.100]

        """

        # get window first position
        i = 0
        window = self[i]

        # yield window while it's valid
        while(window):
            yield window

            # get window next position
            i += 1
            window = self[i]

    def __len__(self):
        """Number of positions

        Equivalent to len([segment for segment in window])

        Returns
        -------
        length : int
            Number of positions taken by the sliding window
            (from start times to end times)

        """
        if np.isinf(self.__end):
            raise ValueError('infinite sliding window.')

        # start looking for last position
        # based on frame closest to the end
        i = self.__closest_frame(self.__end)

        while(self[i]):
            i += 1
        length = i

        return length

    def copy(self):
        """Duplicate sliding window"""
        duration = self.duration
        step = self.step
        start = self.start
        end = self.end
        sliding_window = SlidingWindow(
            duration=duration, step=step, start=start, end=end
        )
        return sliding_window

if __name__ == "__main__":
    import doctest
    doctest.testmod()
