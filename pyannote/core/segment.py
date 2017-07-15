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

"""
#######
Segment
#######

.. plot:: pyplots/segment.py

:class:`pyannote.core.Segment` instances describe temporal fragments (*e.g.* of an audio file). The segment depicted above can be defined like that:

.. ipython::

  In [1]: from pyannote.core import Segment

  In [2]: segment = Segment(start=5, end=15)

  In [3]: print(segment)

It is nothing more than 2-tuples augmented with several useful methods and properties:

.. ipython::

  In [4]: start, end = segment

  In [5]: start

  In [6]: segment.end

  In [7]: segment.duration  # duration (read-only)

  In [8]: segment.middle  # middle (read-only)

  In [9]: segment & Segment(3, 12)  # intersection

  In [10]: segment | Segment(3, 12)  # union

  In [11]: segment.overlaps(3)  # does segment overlap time t=3?

See :class:`pyannote.core.Segment` for the complete reference.
"""


from __future__ import unicode_literals

from collections import namedtuple
import numpy as np

# 1 μs (one microsecond)
SEGMENT_PRECISION = 1e-6


class Segment(namedtuple('Segment', ['start', 'end'])):
    """
    Time interval

    Parameters
    ----------
    start : float
        interval start time, in seconds.
    end : float
        interval end time, in seconds.


    Segments can be compared and sorted using the standard operators:

    >>> Segment(0, 1) == Segment(0, 1.)
    True
    >>> Segment(0, 1) != Segment(3, 4)
    True
    >>> Segment(0, 1) < Segment(2, 3)
    True
    >>> Segment(0, 1) < Segment(0, 2)
    True
    >>> Segment(1, 2) < Segment(0, 3)
    False

    Note
    ----
    A segment is smaller than another segment if one of these two conditions is verified:

      - `segment.start < other_segment.start`
      - `segment.start == other_segment.start` and `segment.end < other_segment.end`

    """

    def __new__(cls, start=0., end=0.):
        return super(Segment, cls).__new__(cls, start, end)

    def __nonzero__(self):
        return self.__bool__()

    def __bool__(self):
        """Emptiness

        >>> if segment:
        ...    # segment is not empty.
        ... else:
        ...    # segment is empty.

        Note
        ----
        A segment is considered empty if its end time is smaller than its
        start time, or its duration is smaller than 1μs.
        """
        return bool((self.end - self.start) > SEGMENT_PRECISION)

    def _get_duration(self):
        return self.end - self.start if self else 0.
    duration = property(fget=_get_duration)
    """Segment duration (read-only)"""

    def _get_middle(self):
        return .5 * (self.start + self.end)
    middle = property(fget=_get_middle)
    """Segment mid-time (read-only)"""

    def __iter__(self):
        """Unpack segment boundaries as float

        >>> segment = Segment(start=1, end=2)
        >>> isinstance(segment.start, int)  # segment.start is int
        True
        >>> start, end = segment
        >>> isinstance(start, float)        # start is float
        True
        """
        yield float(self.start)
        yield float(self.end)

    def copy(self):
        """Get a copy of the segment

        Returns
        -------
        copy : Segment
            Copy of the segment.
        """
        return Segment(start=self.start, end=self.end)

    # ------------------------------------------------------- #
    # Inclusion (in), intersection (&), union (|) and gap (^) #
    # ------------------------------------------------------- #

    def __contains__(self, other):
        """Inclusion

        >>> segment = Segment(start=0, end=10)
        >>> Segment(start=3, end=10) in segment:
        True
        >>> Segment(start=5, end=15) in segment:
        False
        """
        return (self.start <= other.start) and (self.end >= other.end)

    def __and__(self, other):
        """Intersection

        >>> segment = Segment(0, 10)
        >>> other_segment = Segment(5, 15)
        >>> segment & other_segment
        <Segment(5, 10)>

        Note
        ----
        When the intersection is empty, an empty segment is returned:

        >>> segment = Segment(0, 10)
        >>> other_segment = Segment(15, 20)
        >>> intersection = segment & other_segment
        >>> if not intersection:
        ...    # intersection is empty.
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
        intersect : bool
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
        """Check if segment overlaps a given time

        Parameters
        ----------
        t : float
            Time, in seconds.

        Returns
        -------
        overlap: bool
            True if segment overlaps time t, False otherwise.
        """
        return self.start <= t and self.end >= t

    def __or__(self, other):
        """Union

        >>> segment = Segment(0, 10)
        >>> other_segment = Segment(5, 15)
        >>> segment | other_segment
        <Segment(0, 15)>

        Note
        ----
        When a gap exists between the segment, their union covers the gap as well:

        >>> segment = Segment(0, 10)
        >>> other_segment = Segment(15, 20)
        >>> segment | other_segment
        <Segment(0, 20)
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
        """Gap

        >>> segment = Segment(0, 10)
        >>> other_segment = Segment(15, 20)
        >>> segment ^ other_segment
        <Segment(10, 15)

        Note
        ----
        The gap between a segment and an empty segment is not defined.

        >>> segment = Segment(0, 10)
        >>> empty_segment = Segment(11, 11)
        >>> segment ^ empty_segment
        ValueError: The gap between a segment and an empty segment is not defined.
        """

        # if segment is empty, xor is not defined
        if (not self) or (not other):
            raise ValueError(
                'The gap between a segment and an empty segment '
                'is not defined.')

        start = min(self.end, other.end)
        end = max(self.start, other.start)
        return Segment(start=start, end=end)

    def _str_helper(self, seconds):
        from datetime import timedelta
        negative = seconds < 0
        seconds = abs(seconds)
        td = timedelta(seconds=seconds)
        seconds = td.seconds + 86400 * td.days
        microseconds = td.microseconds
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return '%s%02d:%02d:%02d.%03d' % (
            '-' if negative else ' ', hours, minutes,
            seconds, microseconds / 1000)

    def __str__(self):
        """Human-readable representation

        >>> print(Segment(1337, 1337 + 0.42))
        [ 00:22:17.000 -->  00:22:17.420]

        Note
        ----
        Empty segments are printed as "[]"
        """
        if self:
            return '[%s --> %s]' % (self._str_helper(self.start),
                                    self._str_helper(self.end))
        return '[]'

    def pretty(self):
        warnings.warn(
            '"pretty" has been replaced by "__str__"',
            DeprecationWarning)
        return self.__str__()

    def __repr__(self):
        """Computer-readable representation

        >>> Segment(1337, 1337 + 0.42)
        <Segment(1337, 1337.42)>
        """
        return '<Segment(%g, %g)>' % (self.start, self.end)

    def for_json(self):
        """Serialization

        See also
        --------
        :mod:`pyannote.core.json`
        """
        return {'start': self.start, 'end': self.end}

    @classmethod
    def from_json(cls, data):
        """Deserialization

        See also
        --------
        :mod:`pyannote.core.json`
        """
        return cls(start=data['start'], end=data['end'])

    def _repr_png_(self):
        """IPython notebook support

        See also
        --------
        :mod:`pyannote.core.notebook`
        """
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

    def samples(self, from_duration, mode='strict'):
        """Number of frames

        Parameters
        ----------
        from_duration : float
            Duration in seconds.
        mode : {'strict', 'loose', 'center'}
            In 'strict' mode, computes the maximum number of consecutive frames
            that can be fitted into a segment with duration `from_duration`.
            In 'loose' mode, computes the maximum number of consecutive frames
            intersecting a segment with duration `from_duration`.
            In 'center' mode, computes the average number of consecutive frames
            where the first one is centered on the start time and the last one
            is centered on the end time of a segment with duration
            `from_duration`.

        """
        if mode == 'strict':
            return int(np.floor((from_duration - self.duration) / self.step))

        elif mode == 'loose':
            return int(np.floor((from_duration + self.duration) / self.step))

        elif mode == 'center':
            return int(np.rint((from_duration / self.step)))

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

            if fixed is not None:
                n = self.samples(fixed, mode=mode)

            if mode == 'loose':

                # find smallest integer i such that
                # self.start + i x self.step + self.duration >= focus.start
                i_ = (focus.start - self.duration - self.start) / self.step
                i = int(np.ceil(i_))

                if fixed is None:
                    # find largest integer j such that
                    # self.start + j x self.step <= focus.end
                    j_ = (focus.end - self.start) / self.step
                    j = int(np.floor(j_))
                    return np.array(range(i, j + 1), dtype=np.int64)

                else:
                    return np.array(range(i, i + n), dtype=np.int64)


            elif mode == 'strict':

                # find smallest integer i such that
                # self.start + i x self.step >= focus.start
                i_ = (focus.start - self.start) / self.step
                i = int(np.ceil(i_))

                if fixed is None:
                    # find largest integer j such that
                    # self.start + j x self.step + self.duration <= focus.end
                    j_ = (focus.end - self.duration - self.start) / self.step
                    j = int(np.floor(j_))
                    return np.array(range(i, j + 1), dtype=np.int64)

                else:
                    return np.array(range(i, i + n), dtype=np.int64)


            elif mode == 'center':

                # find window position whose center is the closest to focus.start
                i = self.__closest_frame(focus.start)

                if fixed is None:
                    # find window position whose center is the closest to focus.end
                    j = self.__closest_frame(focus.end)
                    return np.array(range(i, j + 1), dtype=np.int64)

                else:
                    return np.array(range(i, i + n), dtype=np.int64)

            else:
                raise ValueError('mode must be "loose", "strict", or "center"')

        elif isinstance(focus, Timeline):
            return np.unique(np.hstack([
                self.crop(s, mode=mode, fixed=fixed) for s in focus.support()]))

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
