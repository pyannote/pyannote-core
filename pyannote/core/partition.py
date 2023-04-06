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
from .base import BaseSegmentation, PureSegmentationMixin
from .segment import Segment
from .utils.types import Support, CropMode

if TYPE_CHECKING:
    pass


# =====================================================================
# Partition class
# =====================================================================


def pairwise(iterable):
    "s -> (s0, s1), (s2, s3), (s4, s5), ..."
    a = iter(iterable)
    return zip(a, a)


class Partition(PureSegmentationMixin, BaseSegmentation):
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
    start: float TODO
    end: float TODO
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
                 start: float = 0.0,
                 end: float = None,
                 uri: str = None):
        segments = list(segments) if segments else []
        if segments is None and end is None:
            raise ValueError("Cannot initialize an empty timeline without and end boundary")
        elif end is None:
            end = max(seg.end for seg in segments)
        elif not segments:
            segments = Segment(start, end)
        super().__init__(uri)
        self.boundaries = Segment(start, end)
        timeline = Timeline(segments)
        if timeline.extent() not in self.boundaries:
            raise ValueError(f"Segments have to be within ({start, end}) bounds")

        # automatically filling in the gaps in the segments
        # TODO: ask about behavior?
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

    def add(self, segment: Segment):
        # TODO: ask about this behavior
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
                         start=self.boundaries.start,
                         end=self.boundaries.end,
                         uri=self.uri)

    def copy(self) -> Self:
        return Partition(self.itersegments(),
                         start=self.boundaries.start,
                         end=self.boundaries.end,
                         uri=self.uri)

    def extent(self) -> Segment:
        return self.boundaries

    def overlapping(self, t: float) -> List[Segment]:
        assert self.boundaries.overlaps(t)
        end = next(self._segments_bounds_set.irange(mininum=t))
        end_idx = self._segments_bounds_set.index(end)
        start = self._segments_bounds_set[end_idx - 1]
        return [Segment(start, end)]

    def crop_iter(self, support: Support, mode: CropMode = 'intersection', returns_mapping: bool = False) -> Iterator[
        Union[Tuple[Segment, Segment], Segment]]:
        pass

    def crop(self, support: Support, mode: CropMode = 'intersection', returns_mapping: bool = False) -> Union[
        Self, Tuple[Self, Dict[Segment, Segment]]]:
        pass

    def duration(self) -> float:
        return self.extent().duration

    def _repr_png_(self):
        pass
