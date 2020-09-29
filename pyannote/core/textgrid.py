#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2014-2020 CNRS

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
# Paul LERNER

"""
##########
Annotation
##########

.. plot:: pyplots/annotation.py

:class:`pyannote.core.Annotation` instances are ordered sets of non-empty
tracks:

  - ordered, because segments are sorted by start time (and end time in case of tie)
  - set, because one cannot add twice the same track
  - non-empty, because one cannot add empty track

A track is a (support, name) pair where `support` is a Segment instance,
and `name` is an additional identifier so that it is possible to add multiple
tracks with the same support.

To define the annotation depicted above:

.. code-block:: ipython

    In [1]: from pyannote.core import Annotation, Segment

    In [6]: annotation = Annotation()
       ...: annotation[Segment(1, 5)] = 'Carol'
       ...: annotation[Segment(6, 8)] = 'Bob'
       ...: annotation[Segment(12, 18)] = 'Carol'
       ...: annotation[Segment(7, 20)] = 'Alice'
       ...:

which is actually a shortcut for

.. code-block:: ipython

    In [6]: annotation = Annotation()
       ...: annotation[Segment(1, 5), '_'] = 'Carol'
       ...: annotation[Segment(6, 8), '_'] = 'Bob'
       ...: annotation[Segment(12, 18), '_'] = 'Carol'
       ...: annotation[Segment(7, 20), '_'] = 'Alice'
       ...:

where all tracks share the same (default) name ``'_'``.

In case two tracks share the same support, use a different track name:

.. code-block:: ipython

    In [6]: annotation = Annotation(uri='my_video_file', modality='speaker')
       ...: annotation[Segment(1, 5), 1] = 'Carol'  # track name = 1
       ...: annotation[Segment(1, 5), 2] = 'Bob'    # track name = 2
       ...: annotation[Segment(12, 18)] = 'Carol'
       ...:

The track name does not have to be unique over the whole set of tracks.

.. note::

  The optional *uri* and *modality* keywords argument can be used to remember
  which document and modality (e.g. speaker or face) it describes.

Several convenient methods are available. Here are a few examples:

.. code-block:: ipython

  In [9]: annotation.labels()   # sorted list of labels
  Out[9]: ['Bob', 'Carol']

  In [10]: annotation.chart()   # label duration chart
  Out[10]: [('Carol', 10), ('Bob', 4)]

  In [11]: list(annotation.itertracks())
  Out[11]: [(<Segment(1, 5)>, 1), (<Segment(1, 5)>, 2), (<Segment(12, 18)>, u'_')]

  In [12]: annotation.label_timeline('Carol')
  Out[12]: <Timeline(uri=my_video_file, segments=[<Segment(1, 5)>, <Segment(12, 18)>])>

See :class:`pyannote.core.Annotation` for the complete reference.
"""
import itertools
from collections import defaultdict
from pathlib import Path
from typing import Optional, Dict, Union, Iterable, List, Set, TextIO, Tuple, Iterator, Text

import numpy as np
from sortedcontainers import SortedDict

from pyannote.core import Annotation
from . import PYANNOTE_URI, PYANNOTE_MODALITY, \
    PYANNOTE_SEGMENT, PYANNOTE_TRACK, PYANNOTE_LABEL
from .json import PYANNOTE_JSON, PYANNOTE_JSON_CONTENT
from .segment import Segment
from .timeline import Timeline
from .utils.generators import string_generator, int_generator
from .utils.types import Label, Key, Support, LabelGenerator, TierName, CropMode


class PraatTier:

    def __init__(self, name: str = None, uri: str = None):
        self.name = name
        self.uri = uri
        self._segments = SortedDict()

        self._needs_labels_update = False
        self._labels_index: Dict[str, Timeline] = dict()

        self._needs_timeline_update = False
        self._timeline = Timeline()

    def __setitem__(self, key: Segment, value: str):
        self._segments.irange()
        self._needs_timeline_update = True
        self._needs_labels_update = True

    def __getitem__(self, item: Segment) -> str:
        return self._segments[item]

    def __delitem__(self, item: Segment):
        del self._segments[item]
        self._needs_timeline_update = True
        self._needs_labels_update = True

    def get_index(self, k: int) -> Tuple[Segment, str]:
        """
        # TODO
        :param k:
        :return:
        """
        return self._segments.peekitem(k)

    def get_timeline(self, copy: bool = False) -> Timeline:
        pass # TODO

    def update(self, tier: 'PraatTier') -> 'PraatTier':
        """Add every segments of an existing tier (in place)

        Parameters
        ----------
        tier : PraatTier
            Tier whose segments and their annotations are being added

        Returns
        -------
        self : PraatTier
            Updated tier

        Note
        ----
        Only segments that do not already exist will be added, as a timeline is
        meant to be a **set** of segments (not a list).

        """
        pass  # TODO

    def __len__(self):
        """Number of segments in the tier

        >>> len(tier)  # tier contains three segments
        3
        """
        return len(self._segments)

    def __nonzero__(self):
        return self.__bool__()

    def __bool__(self):
        """Emptiness

        >>> if tier:
        ...    # timeline is empty
        ... else:
        ...    # timeline is not empty
        """
        return len(self._segments) > 0

    def __iter__(self) -> Iterable[Segment, str]:
        """Iterate over segments (in chronological order)

        >>> for segment, annotation in tier:
        ...     # do something with the segment

        See also
        --------
        :class:`pyannote.core.Segment` describes how segments are sorted.
        """
        return iter(self._segments.items())

    def __eq__(self, other: 'PraatTier'):
        """Equality

        Two PraatTiers are equal if and only if their segments and their annotations are equal.

        # TODO : doc
        >>> timeline1 = Timeline([Segment(0, 1), Segment(2, 3)])
        >>> timeline2 = Timeline([Segment(2, 3), Segment(0, 1)])
        >>> timeline3 = Timeline([Segment(2, 3)])
        >>> timeline1 == timeline2
        True
        >>> timeline1 == timeline3
        False
        """
        return self._segments == other._segments

    def __ne__(self, other: 'PraatTier'):
        """Inequality"""
        return self._segments != other._segments

    def __or__(self, timeline: 'Timeline') -> 'Timeline':
        return self.union(timeline)

    def co_iter(self, other: 'Timeline') -> Iterator[Tuple[Segment, Segment]]:
        """Iterate over pairs of intersecting segments

        >>> timeline1 = Timeline([Segment(0, 2), Segment(1, 2), Segment(3, 4)])
        >>> timeline2 = Timeline([Segment(1, 3), Segment(3, 5)])
        >>> for segment1, segment2 in timeline1.co_iter(timeline2):
        ...     print(segment1, segment2)
        (<Segment(0, 2)>, <Segment(1, 3)>)
        (<Segment(1, 2)>, <Segment(1, 3)>)
        (<Segment(3, 4)>, <Segment(3, 5)>)

        Parameters
        ----------
        other : Timeline
            Second timeline

        Returns
        -------
        iterable : (Segment, Segment) iterable
            Yields pairs of intersecting segments in chronological order.
        """

        for segment in self.segments_list_:

            # iterate over segments that starts before 'segment' ends
            temp = Segment(start=segment.end, end=segment.end)
            for other_segment in other.segments_list_.irange(maximum=temp):
                if segment.intersects(other_segment):
                    yield segment, other_segment

    def crop_iter(self,
                  support: Support,
                  mode: CropMode = 'intersection',
                  returns_mapping: bool = False) \
            -> Iterator[Union[Tuple[Segment, Segment], Segment]]:
        """Like `crop` but returns a segment iterator instead

        See also
        --------
        :func:`pyannote.core.Timeline.crop`
        """

        if mode not in {'loose', 'strict', 'intersection'}:
            raise ValueError("Mode must be one of 'loose', 'strict', or "
                             "'intersection'.")

        if not isinstance(support, (Segment, Timeline)):
            raise TypeError("Support must be a Segment or a Timeline.")

        if isinstance(support, Segment):
            # corner case where "support" is empty
            if support:
                segments = [support]
            else:
                segments = []

            support = Timeline(segments=segments, uri=self.uri)
            for yielded in self.crop_iter(support, mode=mode,
                                          returns_mapping=returns_mapping):
                yield yielded
            return

        # if 'support' is a `Timeline`, we use its support
        support = support.support()

        # loose mode
        if mode == 'loose':
            for segment, _ in self.co_iter(support):
                yield segment
            return

        # strict mode
        if mode == 'strict':
            for segment, other_segment in self.co_iter(support):
                if segment in other_segment:
                    yield segment
            return

        # intersection mode
        for segment, other_segment in self.co_iter(support):
            mapped_to = segment & other_segment
            if not mapped_to:
                continue
            if returns_mapping:
                yield segment, mapped_to
            else:
                yield mapped_to

    def crop(self,
             support: Support,
             mode: CropMode = 'intersection',
             returns_mapping: bool = False) \
            -> 'PraatTier':
        """Crop timeline to new support

        Parameters
        ----------
        support : Segment or Timeline
            If `support` is a `Timeline`, its support is used.
        mode : {'strict', 'loose', 'intersection'}, optional
            Controls how segments that are not fully included in `support` are
            handled. 'strict' mode only keeps fully included segments. 'loose'
            mode keeps any intersecting segment. 'intersection' mode keeps any
            intersecting segment but replace them by their actual intersection.
        returns_mapping : bool, optional
            In 'intersection' mode, return a dictionary whose keys are segments
            of the cropped timeline, and values are list of the original
            segments that were cropped. Defaults to False.

        Returns
        -------
        cropped : Timeline
            Cropped timeline
        mapping : dict
            When 'returns_mapping' is True, dictionary whose keys are segments
            of 'cropped', and values are lists of corresponding original
            segments.

        Examples
        --------

        >>> timeline = Timeline([Segment(0, 2), Segment(1, 2), Segment(3, 4)])
        >>> timeline.crop(Segment(1, 3))
        <Timeline(uri=None, segments=[<Segment(1, 2)>])>

        >>> timeline.crop(Segment(1, 3), mode='loose')
        <Timeline(uri=None, segments=[<Segment(0, 2)>, <Segment(1, 2)>])>

        >>> timeline.crop(Segment(1, 3), mode='strict')
        <Timeline(uri=None, segments=[<Segment(1, 2)>])>

        >>> cropped, mapping = timeline.crop(Segment(1, 3), returns_mapping=True)
        >>> print(mapping)
        {<Segment(1, 2)>: [<Segment(0, 2)>, <Segment(1, 2)>]}

        """

        if mode == 'intersection' and returns_mapping:
            segments, mapping = [], {}
            for segment, mapped_to in self.crop_iter(support,
                                                     mode='intersection',
                                                     returns_mapping=True):
                segments.append(mapped_to)
                mapping[mapped_to] = mapping.get(mapped_to, list()) + [segment]
            return Timeline(segments=segments, uri=self.uri), mapping

        return Timeline(segments=self.crop_iter(support, mode=mode),
                        uri=self.uri)

    def overlapping(self, t: float) -> List[Segment]:
        """Get list of segments overlapping `t`

        Parameters
        ----------
        t : float
            Timestamp, in seconds.

        Returns
        -------
        segments : list
            List of all segments of timeline containing time t
        """
        return list(self.overlapping_iter(t))

    def overlapping_iter(self, t: float) -> Iterator[Segment]:
        """Like `overlapping` but returns a segment iterator instead

        See also
        --------
        :func:`pyannote.core.Timeline.overlapping`
        """
        segment = Segment(start=t, end=t)
        for segment in self.segments_list_.irange(maximum=segment):
            if segment.overlaps(t):
                yield segment

    def __str__(self):
        """Human-readable representation

        >>> timeline = Timeline(segments=[Segment(0, 10), Segment(1, 13.37)])
        >>> print(timeline)
        [[ 00:00:00.000 -->  00:00:10.000]
         [ 00:00:01.000 -->  00:00:13.370]]

        """

        n = len(self.segments_list_)
        string = "["
        for i, segment in enumerate(self.segments_list_):
            string += str(segment)
            string += "\n " if i + 1 < n else ""
        string += "]"
        return string

    def __repr__(self):
        """Computer-readable representation

        >>> Timeline(segments=[Segment(0, 10), Segment(1, 13.37)])
        <Timeline(uri=None, segments=[<Segment(0, 10)>, <Segment(1, 13.37)>])>

        """

        return "<Timeline(uri=%s, segments=%s)>" % (self.uri,
                                                    list(self.segments_list_))

    def __contains__(self, included: Union[Segment, 'Timeline']):
        """Inclusion

        Check whether every segment of `included` does exist in timeline.

        Parameters
        ----------
        included : Segment or Timeline
            Segment or timeline being checked for inclusion

        Returns
        -------
        contains : bool
            True if every segment in `included` exists in timeline,
            False otherwise

        Examples
        --------
        >>> timeline1 = Timeline(segments=[Segment(0, 10), Segment(1, 13.37)])
        >>> timeline2 = Timeline(segments=[Segment(0, 10)])
        >>> timeline1 in timeline2
        False
        >>> timeline2 in timeline1
        >>> Segment(1, 13.37) in timeline1
        True

        """

        if isinstance(included, Segment):
            return included in self.segments_set_

        elif isinstance(included, Timeline):
            return self.segments_set_.issuperset(included.segments_set_)

        else:
            raise TypeError(
                'Checking for inclusion only supports Segment and '
                'Timeline instances')

    def empty(self) -> 'PraatTier':
        """Return an empty copy

        Returns
        -------
        empty : PraatTier
            Empty timeline using the same 'uri' attribute.

        """
        return PraatTier(self.name, uri=self.uri)

    def covers(self, other: Union[Timeline, 'PraatTier']) -> bool:
        """Check whether other timeline  is fully covered by the timeline

        Parameter
        ---------
        other : Timeline
            Second timeline

        Returns
        -------
        covers : bool
            True if timeline covers "other" timeline entirely. False if at least
            one segment of "other" is not fully covered by timeline
        """

        # compute gaps within "other" extent
        # this is where we should look for possible faulty segments
        gaps = self.gaps(support=other.extent())

        # if at least one gap intersects with a segment from "other",
        # "self" does not cover "other" entirely --> return False
        for _ in gaps.co_iter(other):
            return False

        # if no gap intersects with a segment from "other",
        # "self" covers "other" entirely --> return True
        return True

    def copy(self, segment_func: Optional[Callable[[Segment], Segment]] = None) \
            -> 'Timeline':
        """Get a copy of the timeline

        If `segment_func` is provided, it is applied to each segment first.

        Parameters
        ----------
        segment_func : callable, optional
            Callable that takes a segment as input, and returns a segment.
            Defaults to identity function (segment_func(segment) = segment)

        Returns
        -------
        timeline : Timeline
            Copy of the timeline

        """

        # if segment_func is not provided
        # just add every segment
        if segment_func is None:
            return Timeline(segments=self.segments_list_, uri=self.uri)

        # if is provided
        # apply it to each segment before adding them
        return Timeline(segments=[segment_func(s) for s in self.segments_list_],
                        uri=self.uri)

    def extent(self) -> Segment:
        """Extent

        The extent of a timeline is the segment of minimum duration that
        contains every segments of the timeline. It is unique, by definition.
        The extent of an empty timeline is an empty segment.

        A picture is worth a thousand words::

            timeline
            |------|    |------|     |----|
              |--|    |-----|     |----------|

            timeline.extent()
            |--------------------------------|

        Returns
        -------
        extent : Segment
            Timeline extent

        Examples
        --------
        >>> timeline = Timeline(segments=[Segment(0, 1), Segment(9, 10)])
        >>> timeline.extent()
        <Segment(0, 10)>

        """
        if self.segments_set_:
            segments_boundaries_ = self.segments_boundaries_
            start = segments_boundaries_[0]
            end = segments_boundaries_[-1]
            return Segment(start=start, end=end)
        else:
            import numpy as np
            return Segment(start=np.inf, end=-np.inf)

    def support_iter(self, collar: float = 0.) -> Iterator[Segment]:
        """Like `support` but returns a segment generator instead

        See also
        --------
        :func:`pyannote.core.Timeline.support`
        """

        # The support of an empty timeline is an empty timeline.
        if not self:
            return

        # Principle:
        #   * gather all segments with no gap between them
        #   * add one segment per resulting group (their union |)
        # Note:
        #   Since segments are kept sorted internally,
        #   there is no need to perform an exhaustive segment clustering.
        #   We just have to consider them in their natural order.

        # Initialize new support segment
        # as very first segment of the timeline
        new_segment = self.segments_list_[0]

        for segment in self:

            # If there is no gap between new support segment and next segment
            # OR there is a gap with duration < collar seconds,
            possible_gap = segment ^ new_segment
            if not possible_gap or possible_gap.duration < collar:
                # Extend new support segment using next segment
                new_segment |= segment

            # If there actually is a gap and the gap duration >= collar
            # seconds,
            else:
                yield new_segment

                # Initialize new support segment as next segment
                # (right after the gap)
                new_segment = segment

        # Add new segment to the timeline support
        yield new_segment

    def support(self, collar: float = 0.) -> 'Timeline':
        """Timeline support

        The support of a timeline is the timeline with the minimum number of
        segments with exactly the same time span as the original timeline. It
        is (by definition) unique and does not contain any overlapping
        segments.

        A picture is worth a thousand words::

            collar
            |---|

            timeline
            |------|    |------|      |----|
              |--|    |-----|      |----------|

            timeline.support()
            |------|  |--------|   |----------|

            timeline.support(collar)
            |------------------|   |----------|

        Parameters
        ----------
        collar : float, optional
            Merge separated by less than `collar` seconds. This is why there
            are only two segments in the final timeline in the above figure.
            Defaults to 0.

        Returns
        -------
        support : Timeline
            Timeline support
        """
        return Timeline(segments=self.support_iter(collar), uri=self.uri)

    def duration(self) -> float:
        """Timeline duration

        The timeline duration is the sum of the durations of the segments
        in the timeline support.

        Returns
        -------
        duration : float
            Duration of timeline support, in seconds.
        """

        # The timeline duration is the sum of the durations
        # of the segments in the timeline support.
        return sum(s.duration for s in self.support_iter())

    def gaps_iter(self, support: Optional[Support] = None) -> Iterator[Segment]:
        """Like `gaps` but returns a segment generator instead

        See also
        --------
        :func:`pyannote.core.Timeline.gaps`

        """

        if support is None:
            support = self.extent()

        if not isinstance(support, (Segment, Timeline)):
            raise TypeError("unsupported operand type(s) for -':"
                            "%s and Timeline." % type(support).__name__)

        # segment support
        if isinstance(support, Segment):

            # `end` is meant to store the end time of former segment
            # initialize it with beginning of provided segment `support`
            end = support.start

            # support on the intersection of timeline and provided segment
            for segment in self.crop(support, mode='intersection').support():

                # add gap between each pair of consecutive segments
                # if there is no gap, segment is empty, therefore not added
                gap = Segment(start=end, end=segment.start)
                if gap:
                    yield gap

                # keep track of the end of former segment
                end = segment.end

            # add final gap (if not empty)
            gap = Segment(start=end, end=support.end)
            if gap:
                yield gap

        # timeline support
        elif isinstance(support, Timeline):

            # yield gaps for every segment in support of provided timeline
            for segment in support.support():
                for gap in self.gaps_iter(support=segment):
                    yield gap

    def gaps(self, support: Optional[Support] = None) -> 'Timeline':
        """Gaps

        A picture is worth a thousand words::

            tier
            |------|    |------|     |----|

            timeline.gaps()
                   |--|        |----|

        Parameters
        ----------
        support : None, Segment or Timeline
            Support in which gaps are looked for. Defaults to timeline extent

        Returns
        -------
        gaps : Timeline
            Timeline made of all gaps from original timeline, and delimited
            by provided support

        See also
        --------
        :func:`pyannote.core.Timeline.extent`

        """
        return Timeline(segments=self.gaps_iter(support=support),
                        uri=self.uri)

    def segmentation(self) -> 'Timeline':
        """Segmentation

        Create the unique timeline with same support and same set of segment
        boundaries as original timeline, but with no overlapping segments.

        A picture is worth a thousand words::

            timeline
            |------|    |------|     |----|
              |--|    |-----|     |----------|

            timeline.segmentation()
            |-|--|-|  |-|---|--|  |--|----|--|

        Returns
        -------
        timeline : Timeline
            (unique) timeline with same support and same set of segment
            boundaries as original timeline, but with no overlapping segments.
        """
        # COMPLEXITY: O(n)
        support = self.support()

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

        if len(timestamps) == 0:
            return Timeline(uri=self.uri)

        segments = []
        start = timestamps[0]
        for end in timestamps[1:]:
            # only add segments that are covered by original timeline
            segment = Segment(start=start, end=end)
            if segment and support.overlapping(segment.middle):
                segments.append(segment)
            # next segment...
            start = end

        return Timeline(segments=segments, uri=self.uri)

    def to_annotation(self, modality: Optional[str] = None) -> 'Annotation':
        """Turn tier into an annotation

        Each segment is labeled by a unique label.

        Parameters
        ----------
        generator : 'string', 'int', or iterable, optional
            If 'string' (default) generate string labels. If 'int', generate
            integer labels. If iterable, use it to generate labels.
        modality : str, optional

        Returns
        -------
        annotation : Annotation
            Annotation
        """

        from .annotation import Annotation
        annotation = Annotation(uri=self.uri, modality=modality)
        if generator == 'string':
            from .utils.generators import string_generator
            generator = string_generator()
        elif generator == 'int':
            from .utils.generators import int_generator
            generator = int_generator()

        for segment in self:
            annotation[segment] = next(generator)

        return annotation


class PraatTextGrid:
    """Tiered Annotation. Implementation of Praat's TextGrid file structure

    Parameters
    ----------
    uri : string, optional
        name of annotated resource (e.g. audio or video file)
    modality : string, optional
        name of annotated modality

    Returns
    -------
    annotation : Annotation
        New annotation

    """

    def __init__(self, uri: Optional[str] = None):

        self._uri: Optional[str] = uri
        self.modality: Optional[str] = modality

        # sorted dictionary
        # values: {tiername: tier} dictionary
        self._tiers: Dict[TierName, PraatTier] = SortedDict()

        # timeline meant to store all annotated segments
        self._timeline: Timeline = None
        self._timelineNeedsUpdate: bool = True

    @property
    def uri(self):
        return self._uri

    @uri.setter
    def uri(self, uri: str):
        # update uri for all internal timelines
        timeline = self.get_timeline(copy=False)
        timeline.uri = uri
        self._uri = uri

    @property
    def tiers(self) -> List[TierName]:
        return list(self._tiers.keys())

    @property
    def tiers_nb(self):
        return len(self._tiers)

    def __len__(self):
        """Number of segments

        >>> len(textgrid)  # textgrid contains 10 segments
        10
        """
        return sum(len(tier) for tier in self._tiers.values())

    def __nonzero__(self):
        return self.__bool__()

    def __bool__(self):
        """Emptiness

        >>> if annotation:
        ...    # annotation is empty
        ... else:
        ...    # annotation is not empty
        """
        return len(self) > 0

    def itersegments(self):
        """Iterate over segments (in chronological order)

        >>> for segment in annotation.itersegments():
        ...     # do something with the segment

        See also
        --------
        :class:`pyannote.core.Segment` describes how segments are sorted.
        """
        return iter(self._timeline)

    def __iter__(self):
        return iter(self._tiers.items())

    def _update_timeline(self):
        segments = list(itertools.chain.from_iterable(self._tiers.keys()))
        self._timeline = Timeline(segments=segments, uri=self.uri)
        self._timelineNeedsUpdate = False

    def get_timeline(self, copy: bool = True) -> Timeline:
        """Get timeline made of all annotated segments

        Parameters
        ----------
        copy : bool, optional
            Defaults (True) to returning a copy of the internal timeline.
            Set to False to return the actual internal timeline (faster).

        Returns
        -------
        timeline : Timeline
            Timeline made of all annotated segments.

        Note
        ----
        In case copy is set to False, be careful **not** to modify the returned
        timeline, as it may lead to weird subsequent behavior of the annotation
        instance.

        """
        if self._timelineNeedsUpdate:
            self._update_timeline()
        if copy:
            return self._timeline.copy()
        return self._timeline

    def __eq__(self, other: 'PraatTextGrid'):
        """Equality

        >>> annotation == other

        Two annotations are equal if and only if their tracks and associated
        labels are equal.
        """
        pairOfTracks = itertools.zip_longest(
            self.itertracks(yield_label=True),
            other.itertracks(yield_label=True))
        return all(t1 == t2 for t1, t2 in pairOfTracks)

    def __ne__(self, other: 'PraatTextGrid'):
        """Inequality"""
        pairOfTracks = itertools.zip_longest(
            self.itertracks(yield_label=True),
            other.itertracks(yield_label=True))

        return any(t1 != t2 for t1, t2 in pairOfTracks)

    def __contains__(self, included: Union[Segment, Timeline]):
        """Inclusion

        Check whether every segment of `included` does exist in annotation.

        Parameters
        ----------
        included : Segment or Timeline
            Segment or timeline being checked for inclusion

        Returns
        -------
        contains : bool
            True if every segment in `included` exists in timeline,
            False otherwise

        """
        return included in self.get_timeline(copy=False)

    def to_textgrid(self, file: Union[str, Path, TextIO]):
        pass

    def to_annotation(self, modality: Optional[str] = None) -> Annotation:
        """Convert to an annotation object. The new annotation's labels
        are the tier names of each segments. In short, the segment's
        # TODO : visual example

        Parameters
        ----------
        modality: optional str

        Returns
        -------
        annotation : Annotation
            A new Annotation Object

        Note
        ----
        If you want to convert part of a `PraatTextGrid` to an `Annotation` object
        while keeping the segment's labels, you can use the tier's
        :func:`~pyannote.textgrid.PraatTier.to_annotation`
        """
        annotation = Annotation(uri=self.uri, modality=modality)
        for tier_name, tier in self._tiers.items():
            for segment, _ in tier:
                annotation[segment] = tier_name
        return annotation

    def crop(self, support: Support, mode: CropMode = 'intersection') \
            -> 'PraatTextGrid':
        """Crop textgrid to new support

        Parameters
        ----------
        support : Segment or Timeline
            If `support` is a `Timeline`, its support is used.
        mode : {'strict', 'loose', 'intersection'}, optional
            Controls how segments that are not fully included in `support` are
            handled. 'strict' mode only keeps fully included segments. 'loose'
            mode keeps any intersecting segment. 'intersection' mode keeps any
            intersecting segment but replace them by their actual intersection.

        Returns
        -------
        cropped : PraatTextGrid
            Cropped textgrid
        """
        new_tg = PraatTextGrid(self.uri)
        for tier_name, tier in self._tiers.items():
            new_tg._tiers[tier_name] = tier.crop(support)
        return new_tg

    def copy(self) -> 'PraatTextGrid':
        """Get a copy of the annotation

        Returns
        -------
        annotation : PraatTextGrid
            Copy of the textgrid
        """

        # create new empty annotation
        pass

    def __str__(self):
        """Human-friendly representation"""
        # TODO: use pandas.DataFrame
        return "\n".join(["%s %s %s" % (s, t, l)
                          for s, t, l in self.itertracks(yield_label=True)])

    def __delitem__(self, key: TierName):
        """Delete a tier
        # TODO : doc
        """
        del self._tiers[key]

    def __getitem__(self, key: TierName) -> PraatTier:
        """Get a tier

        >>> praat_tier = annotation[tiername]

        """

        return self._tiers[key]

    def __setitem__(self, key: Key, label: Label):
        """Add new or update existing track

        >>> annotation[segment, track] = label

        If (segment, track) does not exist, it is added.
        If (segment, track) already exists, it is updated.

        Note
        ----
        ``annotation[segment] = label`` is equivalent to ``annotation[segment, '_'] = label``

        Note
        ----
        If `segment` is empty, it does nothing.
        """

        if isinstance(key, Segment):
            key = (key, '_')

        segment, track = key

        # do not add empty track
        if not segment:
            return

        # in case we create a new segment
        # mark timeline as modified
        if segment not in self._tiers:
            self._tiers[segment] = {}
            self._timelineNeedsUpdate = True

        # in case we modify an existing track
        # mark old label as modified
        if track in self._tiers[segment]:
            old_label = self._tiers[segment][track]
            self._labelNeedsUpdate[old_label] = True

        # mark new label as modified
        self._tiers[segment][track] = label
        self._labelNeedsUpdate[label] = True

    def empty(self) -> 'Annotation':
        """Return an empty copy

        Returns
        -------
        empty : Annotation
            Empty annotation using the same 'uri' and 'modality' attributes.

        """
        return self.__class__(uri=self.uri, modality=self.modality)

    def update(self, textgrid: 'PraatTextGrid', copy: bool = False) \
            -> 'PraatTextGrid':
        """Add every track of an existing annotation (in place)

        Parameters
        ----------
        annotation : Annotation
            Annotation whose tracks are being added
        copy : bool, optional
            Return a copy of the annotation. Defaults to updating the
            annotation in-place.

        Returns
        -------
        self : Annotation
            Updated annotation

        Note
        ----
        Existing tracks are updated with the new label.
        """

        result = self.copy() if copy else self

        # TODO speed things up by working directly with annotation internals
        for segment, track, label in annotation.itertracks(yield_label=True):
            result[segment, track] = label

        return result

    def chart(self, percent: bool = False) -> List[Tuple[Label, float]]:
        """Get labels chart (from longest to shortest duration)

        Parameters
        ----------
        percent : bool, optional
            Return list of (label, percentage) tuples.
            Defaults to returning list of (label, duration) tuples.

        Returns
        -------
        chart : list
            List of (label, duration), sorted by duration in decreasing order.
        """

        chart = sorted(((L, self.label_duration(L)) for L in self.labels()),
                       key=lambda x: x[1], reverse=True)

        if percent:
            total = np.sum([duration for _, duration in chart])
            chart = [(label, duration / total) for (label, duration) in chart]

        return chart

    def argmax(self, support: Optional[Support] = None) -> Optional[Label]:
        """Get label with longest duration

        Parameters
        ----------
        support : Segment or Timeline, optional
            Find label with longest duration within provided support.
            Defaults to whole extent.

        Returns
        -------
        label : any existing label or None
            Label with longest intersection

        Examples
        --------
        >>> annotation = Annotation(modality='speaker')
        >>> annotation[Segment(0, 10), 'speaker1'] = 'Alice'
        >>> annotation[Segment(8, 20), 'speaker1'] = 'Bob'
        >>> print "%s is such a talker!" % annotation.argmax()
        Bob is such a talker!
        >>> segment = Segment(22, 23)
        >>> if not annotation.argmax(support):
        ...    print "No label intersecting %s" % segment
        No label intersection [22 --> 23]

        """

        cropped = self
        if support is not None:
            cropped = cropped.crop(support, mode='intersection')

        if not cropped:
            return None

        return max(((_, cropped.label_duration(_)) for _ in cropped.labels()),
                   key=lambda x: x[1])[0]

    def support(self, collar: float = 0.) -> 'PraatTextGrid':
        # TODO
        """Annotation support

        The support of an annotation is an annotation where contiguous tracks
        with same label are merged into one unique covering track.

        A picture is worth a thousand words::

            collar
            |---|

            annotation
            |--A--| |--A--|     |-B-|
              |-B-|    |--C--|     |----B-----|

            annotation.support(collar)
            |------A------|     |------B------|
              |-B-|    |--C--|

        Parameters
        ----------
        collar : float, optional
            Merge tracks with same label and separated by less than `collar`
            seconds. This is why 'A' tracks are merged in above figure.
            Defaults to 0.

        Returns
        -------
        support : Annotation
            Annotation support

        Note
        ----
        Track names are lost in the process.
        """

        generator = string_generator()

        # initialize an empty annotation
        # with same uri and modality as original
        support = self.empty()
        for label in self.labels():

            # get timeline for current label
            timeline = self.label_timeline(label, copy=True)

            # fill the gaps shorter than collar
            timeline = timeline.support(collar)

            # reconstruct annotation with merged tracks
            for segment in timeline.support():
                support[segment, next(generator)] = label

        return support

    def co_iter(self, other: 'Annotation') \
            -> Iterator[Tuple[Tuple[Segment, TierName],
                              Tuple[Segment, TierName]]
            ]:
        """Iterate over pairs of intersecting tracks

        Parameters
        ----------
        other : Annotation
            Second annotation

        Returns
        -------
        iterable : (Segment, object), (Segment, object) iterable
            Yields pairs of intersecting tracks, in chronological (then
            alphabetical) order.

        See also
        --------
        :func:`~pyannote.core.Timeline.co_iter`

        """
        timeline = self.get_timeline(copy=False)
        other_timeline = other.get_timeline(copy=False)
        for s, S in timeline.co_iter(other_timeline):
            tracks = sorted(self.get_tracks(s), key=str)
            other_tracks = sorted(other.get_tracks(S), key=str)
            for t, T in itertools.product(tracks, other_tracks):
                yield (s, t), (S, T)

    def __mul__(self, other: 'Annotation') -> np.ndarray:
        """Cooccurrence (or confusion) matrix

        >>> matrix = annotation * other

        Parameters
        ----------
        other : Annotation
            Second annotation

        Returns
        -------
        cooccurrence : (n_self, n_other) np.ndarray
            Cooccurrence matrix where `n_self` (resp. `n_other`) is the number
            of labels in `self` (resp. `other`).
        """

        if not isinstance(other, Annotation):
            raise TypeError(
                'computing cooccurrence matrix only works with Annotation '
                'instances.')

        i_labels = self.labels()
        j_labels = other.labels()

        I = {label: i for i, label in enumerate(i_labels)}
        J = {label: j for j, label in enumerate(j_labels)}

        matrix = np.zeros((len(I), len(J)))

        # iterate over intersecting tracks and accumulate durations
        for (segment, track), (other_segment, other_track) in self.co_iter(other):
            i = I[self[segment, track]]
            j = J[other[other_segment, other_track]]
            duration = (segment & other_segment).duration
            matrix[i, j] += duration

        return matrix

    def _repr_png(self):
        """IPython notebook support

        See also
        --------
        :mod:`pyannote.core.notebook`
        """

        from .notebook import repr_annotation
        return repr_annotation(self)
