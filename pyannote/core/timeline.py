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

import warnings

from . import PYANNOTE_URI, PYANNOTE_SEGMENT
from banyan import SortedSet
from .interval_tree import TimelineUpdator
from .segment import Segment
from .json import PYANNOTE_JSON, PYANNOTE_JSON_CONTENT

# ignore Banyan warning
warnings.filterwarnings(
    'ignore', 'Key-type optimization',
    Warning, 'pyannote.core.timeline'
)
warnings.filterwarnings(
    'ignore', 'Key-type optimization',
    Warning, 'banyan'
)


# =====================================================================
# Timeline class
# =====================================================================


class Timeline(object):
    """
    Ordered set of segments.

    A timeline can be seen as an ordered set of non-empty segments (Segment).
    Segments can overlap -- though adding an already exisiting segment to a
    timeline does nothing.

    Parameters
    ----------
    segments : Segment iterator, optional
        initial set of segments
    uri : string, optional
        name of segmented resource

    Returns
    -------
    timeline : Timeline
        New timeline

    Examples
    --------
    Create a new empty timeline

        >>> timeline = Timeline()
        >>> if not timeline:
        ...    print "Timeline is empty."
        Timeline is empty.

    Add one segment (+=)

        >>> segment = Segment(0, 1)
        >>> timeline.add(segment)
        >>> if len(timeline) == 1:
        ...    print "Timeline contains only one segment."
        Timeline contains only one segment.

    Add all segments from another timeline

        >>> other_timeline = Timeline([Segment(0.5, 3), Segment(6, 8)])
        >>> timeline.update(other_timeline)

    Get timeline extent, coverage & duration

        >>> extent = timeline.extent()
        >>> print extent
        [0 --> 8]
        >>> coverage = timeline.coverage()
        >>> print coverage
        [
           [0 --> 3]
           [6 --> 8]
        ]
        >>> duration = timeline.duration()
        >>> print "Timeline covers a total of %g seconds." % duration
        Timeline covers a total of 5 seconds.

    Iterate over (sorted) timeline segments

        >>> for segment in timeline:
        ...    print segment
        [0 --> 1]
        [0.5 --> 3]
        [6 --> 8]

    Segmentation

        >>> segmentation = timeline.segmentation()
        >>> print segmentation
        [
           [0 --> 0.5]
           [0.5 --> 1]
           [1 --> 3]
           [6 --> 8]
        ]

    Gaps

        >>> timeline = timeline.copy()
        >>> print timeline
        [
           [0 --> 1]
           [0.5 --> 3]
           [6 --> 8]
        ]
        >>> print timeline.gaps()
        [
           [3 --> 6]
        ]
        >>> segment = Segment(0, 10)
        >>> print timeline.gaps(segment)
        [
           [3 --> 6]
           [8 --> 10]
        ]

    """

    @classmethod
    def from_df(cls, df, uri=None):
        segments = list(df[PYANNOTE_SEGMENT])
        timeline = cls(segments=segments, uri=uri)
        return timeline

    def __init__(self, segments=None, uri=None):

        super(Timeline, self).__init__()

        # sorted set of segments (as an augmented red-black tree)
        segments = [s for s in segments if s] if segments else []
        self._segments = SortedSet(items=segments,
                                   key_type=(float, float),
                                   updator=TimelineUpdator)

        # path to (or any identifier of) segmented resource
        self.uri = uri

    def __len__(self):
        return self._segments.length()

    def __bool__(self):
        return self._segments.length() > 0

    def __nonzero__(self):
        return self.__bool__()

    def __iter__(self):
        return iter(self._segments)

    def __getitem__(self, k):
        """Returns kth segment"""
        return self._segments.kth(k)

    def __eq__(self, other):
        return self._segments == other._segments

    def __ne__(self, other):
        return self._segments != other._segments

    def index(self, segment):
        """Index of segment

        Parameter
        ---------
        segment : Segment

        Raises
        ------
        ValueError if the segment is not present
        """
        return self._segments.index(segment)

    def add(self, segment):
        """Add segment"""
        if segment:
            self._segments.add(segment)

    def update(self, timeline):
        """Add `timeline` segments"""
        self._segments.update(timeline._segments)

    def union(self, other):
        """Create new timeline made of union of segments"""
        segments = self._segments.union(other._segments)
        return Timeline(segments=segments, uri=self.uri)

    def co_iter(self, other):
        for segment, other_segment in self._segments.co_iter(other._segments):
            yield segment, other_segment

    def crop(self, other, mode='intersection', mapping=False):
        """Crop timeline

        Parameters
        ----------
        other : `Segment` or `Timeline`

        mode : {'strict', 'loose', 'intersection'}, optional
            In 'strict' mode, only segments fully included in focus coverage
            are kept. In 'loose' mode, any intersecting segment is kept
            unchanged. In 'intersection' mode, only intersecting segments are
            kept and replaced by their actual intersection with the focus.
        mapping : boolean, optional
            [FIXME] When True and mode is 'intersection', also returns the list of
            original corresponding segments.

        Returns
        -------
        cropped : Timeline
            Cropped timeline.
        mapping : dict (if mapping is True and mode is 'intersection')
        """


        if isinstance(other, Segment):
            other = Timeline(segments=[other], uri=self.uri)
            return self.crop(other, mode=mode, mapping=mapping)

        elif isinstance(other, Timeline):

            if mode == 'loose':
                segments = [segment for segment, _ in self.co_iter(other)]
                return Timeline(segments=segments, uri=self.uri)

            elif mode == 'strict':
                segments = [segment
                            for segment, other_segment in self.co_iter(other)
                            if segment in other_segment]
                return Timeline(segments=segments, uri=self.uri)

            elif mode == 'intersection':
                if mapping:
                    mapping = {}
                    for segment, other_segment in self.co_iter(other):
                        inter = segment & other_segment
                        mapping[inter] = mapping.get(inter, list()) + [segment]
                    return Timeline(segments=mapping, uri=self.uri), mapping
                else:
                    segments = [segment & other_segment
                                for segment, other_segment in self.co_iter(other)]
                    return Timeline(segments=segments, uri=self.uri)

            else:
                raise NotImplementedError("unsupported mode: '%s'" % mode)

    def overlapping(self, timestamp):
        """Get list of segments overlapping `timestamp`"""
        return self._segments.overlapping(timestamp)

    def __str__(self):
        """Human-friendly representation"""

        string = "[\n"
        for segment in self._segments:
            string += "   %s\n" % str(segment)
        string += "]"
        return string

    def __repr__(self):
        return "<Timeline(uri=%s, segments=%s)>" % (self.uri,
                                                    list(self._segments))

    def __contains__(self, included):
        """Inclusion

        Use expression 'segment in timeline' or 'other_timeline in timeline'

        Parameters
        ----------
        included : `Segment` or `Timeline`

        Returns
        -------
        contains : bool
            True if every segment in `included` exists in timeline,
            False otherwise

        """

        if isinstance(included, Segment):
            return included in self._segments

        elif isinstance(included, Timeline):
            return self._segments.issuperset(included._segments)

        else:
            raise TypeError()

    def empty(self):
        """Empty copy of a timeline.

        Examples
        --------

            >>> timeline = Timeline(uri="MyVideo.avi")
            >>> timeline += [Segment(0, 1), Segment(2, 3)]
            >>> empty = timeline.empty()
            >>> print empty.uri
            MyVideo.avi
            >>> print empty
            [
            ]

        """
        return Timeline(uri=self.uri)

    def copy(self, segment_func=None):
        """Duplicate timeline.

        If segment_func is provided, apply it to each segment first.

        Parameters
        ----------
        segment_func : function

        Returns
        -------
        timeline : Timeline
            A (possibly modified) copy of the timeline

        Examples
        --------

            >>> timeline = Timeline(uri="MyVideo.avi")
            >>> timeline += [Segment(0, 1), Segment(2, 3)]
            >>> cp = timeline.copy()
            >>> print cp.uri
            MyVideo.avi
            >>> print cp
            [
               [0 --> 1]
               [2 --> 3]
            ]

        """

        # if segment_func is not provided
        # just add every segment
        if segment_func is None:
            return Timeline(segments=self._segments, uri=self.uri)

        # if is provided
        # apply it to each segment before adding them
        else:
            return Timeline(segments=[segment_func(s) for s in self._segments],
                            uri=self.uri)

    def extent(self):
        """Timeline extent

        The extent of a timeline is the segment of minimum duration that
        contains every segments of the timeline. It is unique, by definition.
        The extent of an empty timeline is an empty segment.

        Returns
        -------
        extent : Segment
            Timeline extent

        Examples
        --------

            >>> timeline = Timeline(uri="MyVideo.avi")
            >>> timeline += [Segment(0, 1), Segment(9, 10)]
            >>> print timeline.extent()
            [0 --> 10]

        """
        return self._segments.extent()

    def coverage(self):
        """Timeline coverage

        The coverage of timeline is the timeline with the minimum number of
        segments with exactly the same time span as the original timeline.
        It is (by definition) unique and does not contain any overlapping
        segments.

        Returns
        -------
        coverage : Timeline
            Timeline coverage

        """

        # make sure URI attribute is kept.
        coverage = Timeline(uri=self.uri)

        # The coverage of an empty timeline is an empty timeline.
        if not self:
            return coverage

        # Principle:
        #   * gather all segments with no gap between them
        #   * add one segment per resulting group (their union |)
        # Note:
        #   Since segments are kept sorted internally,
        #   there is no need to perform an exhaustive segment clustering.
        #   We just have to consider them in their natural order.

        # Initialize new coverage segment
        # as very first segment of the timeline
        new_segment = self._segments.kth(0)

        for segment in self:

            # If there is no gap between new coverage segment and next segment,
            if not (segment ^ new_segment):
                # Extend new coverage segment using next segment
                new_segment |= segment

            # If there actually is a gap,
            else:
                # Add new segment to the timeline coverage
                coverage.add(new_segment)
                # Initialize new coverage segment as next segment
                # (right after the gap)
                new_segment = segment

        # Add new segment to the timeline coverage
        coverage.add(new_segment)

        return coverage

    def duration(self):
        """Timeline duration

        Returns
        -------
        duration : float
            Duration of timeline coverage, in seconds.

        """

        # The timeline duration is the sum of the durations
        # of the segments in the timeline coverage.
        return sum([s.duration for s in self.coverage()])

    def gaps(self, focus=None):
        """Timeline gaps

        Parameters
        ----------
        focus : None, Segment or Timeline

        Returns
        -------
        gaps : Timeline
            Timeline made of all gaps from original timeline, and delimited
            by provided segment or timeline.

        Raises
        ------
        TypeError when `focus` is neither None, Segment nor Timeline

        Examples
        --------

        """
        if focus is None:
            focus = self.extent()

        if not isinstance(focus, (Segment, Timeline)):
            raise TypeError("unsupported operand type(s) for -':"
                            "%s and Timeline." % type(focus).__name__)

        # segment focus
        if isinstance(focus, Segment):

            # starts with an empty timeline
            timeline = self.empty()

            # `end` is meant to store the end time of former segment
            # initialize it with beginning of provided segment `focus`
            end = focus.start

            # focus on the intersection of timeline and provided segment
            for segment in self.crop(focus, mode='intersection').coverage():

                # add gap between each pair of consecutive segments
                # if there is no gap, segment is empty, therefore not added
                timeline.add(Segment(start=end, end=segment.start))

                # keep track of the end of former segment
                end = segment.end

            # add final gap (if not empty)
            timeline.add(Segment(start=end, end=focus.end))

        # other_timeline - timeline
        elif isinstance(focus, Timeline):

            # starts with an empty timeline
            timeline = self.empty()

            # add gaps for every segment in coverage of provided timeline
            for segment in focus.coverage():
                timeline.update(self.gaps(focus=segment))

        return timeline

    def segmentation(self):
        """Non-overlapping timeline

        Create the unique timeline with same coverage and same set of segment
        boundaries as original timeline, but with no overlapping segments.

        A picture is worth a thousand words:

            Original timeline:
            |------|    |------|     |----|
              |--|    |-----|     |----------|

            Non-overlapping timeline
            |-|--|-|  |-|---|--|  |--|----|--|

        Returns
        -------
        timeline : Timeline

        Examples
        --------

            >>> timeline = Timeline()
            >>> timeline += [Segment(0, 1), Segment(1, 2), Segment(2,3)]
            >>> timeline += [Segment(2, 4), Segment(6, 7)]
            >>> print timeline.segmentation()
            [
               [0 --> 1]
               [1 --> 2]
               [2 --> 3]
               [3 --> 4]
               [6 --> 7]
            ]

        """
        # COMPLEXITY: O(n)
        coverage = self.coverage()

        # COMPLEXITY: O(n.log n)
        # get all boundaries (sorted)
        # |------|    |------|     |----|
        #   |--|    |-----|     |----------|
        # becomes
        # | |  | |  | |   |  |  |  |    |  |
        timestamps = set([])
        for (start, end) in self:
            timestamps.add(start)
            timestamps.add(end)
        timestamps = sorted(timestamps)

        # create new partition timeline
        # | |  | |  | |   |  |  |  |    |  |
        # becomes
        # |-|--|-|  |-|---|--|  |--|----|--|

        # start with an empty copy
        timeline = Timeline(uri=self.uri)

        if len(timestamps) > 0:
            segments = []
            start = timestamps[0]
            for end in timestamps[1:]:

                # only add segments that are covered by original timeline
                segment = Segment(start=start, end=end)
                if segment and coverage.overlapping(segment.middle):
                    segments.append(segment)
                # next segment...

                start = end

            timeline._segments.update(segments)

        return timeline

    def for_json(self):
        data = {PYANNOTE_JSON: self.__class__.__name__}
        data[PYANNOTE_JSON_CONTENT] = [s.for_json() for s in self]

        if self.uri:
            data[PYANNOTE_URI] = self.uri

        return data

    @classmethod
    def from_json(cls, data):
        uri = data.get(PYANNOTE_URI, None)
        segments = [Segment.from_json(s) for s in data[PYANNOTE_JSON_CONTENT]]
        return cls(segments=segments, uri=uri)

    def _repr_png_(self):
        from .notebook import repr_timeline
        return repr_timeline(self)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
