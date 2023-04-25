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
# Grant JENKS - http://www.grantjenks.com/
# Paul LERNER

"""
########
Timeline
########

.. plot:: pyplots/timeline.py

:class:`pyannote.core.Timeline` instances are ordered sets of non-empty
segments:

  - ordered, because segments are sorted by start time (and end time in case of tie)
  - set, because one cannot add twice the same segment
  - non-empty, because one cannot add empty segments (*i.e.* start >= end)

There are two ways to define the timeline depicted above:

.. code-block:: ipython

  In [25]: from pyannote.core import Timeline, Segment

  In [26]: timeline = Timeline()
     ....: timeline.add(Segment(1, 5))
     ....: timeline.add(Segment(6, 8))
     ....: timeline.add(Segment(12, 18))
     ....: timeline.add(Segment(7, 20))
     ....:

  In [27]: segments = [Segment(1, 5), Segment(6, 8), Segment(12, 18), Segment(7, 20)]
     ....: timeline = Timeline(segments=segments, uri='my_audio_file')  # faster
     ....:

  In [9]: for segment in timeline:
     ...:     print(segment)
     ...:
  [ 00:00:01.000 -->  00:00:05.000]
  [ 00:00:06.000 -->  00:00:08.000]
  [ 00:00:07.000 -->  00:00:20.000]
  [ 00:00:12.000 -->  00:00:18.000]


.. note::

  The optional *uri*  keyword argument can be used to remember which document it describes.

Several convenient methods are available. Here are a few examples:

.. code-block:: ipython

  In [3]: timeline.extent()    # extent
  Out[3]: <Segment(1, 20)>

  In [5]: timeline.support()  # support
  Out[5]: <Timeline(uri=my_audio_file, segments=[<Segment(1, 5)>, <Segment(6, 20)>])>

  In [6]: timeline.duration()  # support duration
  Out[6]: 18


See :class:`pyannote.core.Timeline` for the complete reference.
"""
from typing import (Optional, Iterable, Union, Callable,
                    Tuple, TYPE_CHECKING, Iterator, Dict, List)

from sortedcontainers import SortedDict, SortedList, SortedSet
from typing_extensions import Self

from . import Timeline
from .base import BaseSegmentation, PureSegmentationMixin, ContiguousAnnotationMixin
from .segment import Segment
from .utils.types import Support, CropMode, ContiguousSupport

if TYPE_CHECKING:
    pass


# =====================================================================
# Partition class
# =====================================================================


def pairwise(iterable):
    "s -> (s0, s1), (s2, s3), (s4, s5), ..."
    a = iter(iterable)
    return zip(a, a)


class Partition(PureSegmentationMixin, ContiguousAnnotationMixin, BaseSegmentation):
    """
    Ordered set of segments that are all contiguous.

    It has a start and a end boundary, and its segments form a continuum
    between those two boundaries. Segments can be created by bisecting
    the partition at certain points, and removing a segment amounts to
    removing the bisections.

    Parameters
    ----------
    segments : Segment iterator, optional
        initial set of (non-empty) segments
    boundaries: Segment, optional
    uri : string, optional
        name of segmented resource

    Returns
    -------
    timeline : Timeline
        New timeline
    """

    # TODO: need to reimplement the co_iter function to make it much faster (the whole point of having partitions)
    #  -> if co_iter with another partition, even faster (need to distinguish the case)

    def __init__(self,
                 segments: Optional[Iterable[Segment]] = None,
                 boundaries: Optional[Segment] = None,
                 uri: str = None):
        segments = list(segments) if segments else []
        if not segments and boundaries is None:
            raise ValueError("Cannot initialize an empty Partition without definin boundaries")
        super().__init__(uri)
        self.boundaries = boundaries
        timeline = Timeline(segments)
        if timeline.extent() not in self.boundaries:
            raise ValueError(f"Segments have to be within {boundaries}")

        # automatically filling in the gaps in the segments
        timeline.add(self.boundaries)
        self._segments_bounds_set = SortedSet()
        for (start, end) in timeline:
            self._segments_bounds_set.update(start, end)

    def __len__(self) -> int:
        return len(self._segments_bounds_set) - 1

    def __nonzero__(self):
        return True

    def __bool__(self):
        return True

    def __eq__(self, other: 'Partition'):
        return isinstance(other, Partition) and self._segments_set == other._segments_set

    def __ne__(self, other: 'Partition'):
        return not other == self

    def __iter__(self) -> Iterable[Segment]:
        return self.itersegments()

    def index(self, segment: Segment) -> int:
        return self._segments_bounds_set.index(segment.start)

    def bisect(self, at: float):
        if not self.boundaries.overlaps(at):
            raise ValueError("Cannot bisect outside of partition boundaries")

        self._segments_bounds_set.add(at)

    def fuse(self, at: float):
        try:
            self._segments_bounds_set.remove(at)
        except KeyError:
            raise RuntimeError("Cannot fuse non-existing boundary")

    def add(self, segment: Segment):
        # TODO: fix (check for segment inclusion)
        if len(list(self.co_iter(segment))) > 1:
            raise ValueError("Segment overlaps a boundary")
        self.bisect(segment.start)
        self.bisect(segment.end)

    def remove(self, segment: Segment):
        if not (set(segment) & self._segments_bounds_set):
            raise KeyError(f"Segment {segment} not in partition")
        self._segments_bounds_set.difference_update(segment)

    def itersegments(self):
        for (start, end) in pairwise(self._segments_bounds_set):
            yield Segment(start, end)

    def get_timeline(self) -> 'Timeline':
        return Timeline(self.itersegments(), uri=self.uri)

    def update(self, other: 'Partition') -> 'Partition':
        assert other.boundaries in self.boundaries
        self._segments_bounds_set |= other._segments_bounds_set

    def __str__(self):
        pass

    def __repr__(self):
        pass

    def empty(self) -> Self:
        return Partition(None,
                         boundaries=self.boundaries,
                         uri=self.uri)

    def copy(self) -> Self:
        return Partition(self.itersegments(),
                         boundaries=self.boundaries,
                         uri=self.uri)

    def extent(self) -> Segment:
        return self.boundaries

    def overlapping(self, t: float) -> List[Segment]:
        assert self.boundaries.overlaps(t)
        end = next(self._segments_bounds_set.irange(mininum=t))
        end_idx = self._segments_bounds_set.index(end)
        start = self._segments_bounds_set[end_idx - 1]
        return [Segment(start, end)]

    def crop_iter(self, support: ContiguousSupport, mode: CropMode = 'intersection', returns_mapping: bool = False) \
            -> Iterator[Union[Tuple[Segment, Segment], Segment]]:
        # TODO: check algo when boundaries match
        if not isinstance(support, (Segment, ContiguousAnnotationMixin)):
            raise ValueError(f"Only contiguous supports are allowed for cropping a {self.__class__.__name__}.")

        if not isinstance(support, Segment):
            support = support.extent()
        if self.extent() in support:
            return self.itersegments()

        cropped_boundaries = SortedSet(self._segments_bounds_set.irange(minimum=support.start,
                                                                        maximum=support.end,
                                                                        inclusive=(False, False)))

        # first, yielding the first "cut" segment depending on mode
        if support.start > self.extent().start:
            idx_start = self._segments_bounds_set.index(cropped_boundaries[0])
            first_seg = Segment(start=self._segments_bounds_set[idx_start - 1], end=self._segments_bounds_set[idx_start])
            if mode == "intersection":
                mapped_to = Segment(start=support.start, end=first_seg.end)
                if returns_mapping:
                    yield first_seg, mapped_to
                else:
                    yield mapped_to
            elif mode == "loose":
                yield first_seg

        # then, yielding "untouched" segments
        for (start, end) in pairwise(cropped_boundaries):
            seg = Segment(start, end)
            if returns_mapping:
                yield seg, seg
            else:
                yield seg

        # finally, yielding the last "cut" segment depending on mode
        if support.end < self.extent().end:
            idx_end = self._segments_bounds_set.index(cropped_boundaries[0])
            last_seg = Segment(start=self._segments_bounds_set[idx_end], end=self._segments_bounds_set[idx_end + 1])
            if mode == "intersection":
                mapped_to = Segment(start=last_seg.start, end=support.end)
                if returns_mapping:
                    yield last_seg, mapped_to
                else:
                    yield last_seg
            elif mode == "loose":
                yield last_seg

    def crop(self, support: ContiguousSupport, mode: CropMode = 'intersection', returns_mapping: bool = False) \
            -> Union[Self, Tuple[Self, Dict[Segment, Segment]]]:
        if mode == 'intersection' and returns_mapping:
            segments, mapping = [], {}
            for segment, mapped_to in self.crop_iter(support,
                                                     mode='intersection',
                                                     returns_mapping=True):
                segments.append(segment)
                mapping[mapped_to] = [segment]
            return Partition(segments=segments, uri=self.uri), mapping

        return Partition(segments=self.crop_iter(support, mode=mode), uri=self.uri)

    def duration(self) -> float:
        return self.extent().duration

    def _repr_png_(self):
        pass
