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
import warnings
from typing import (Optional, Iterable, List, Union, Callable,
                    TextIO, Tuple, TYPE_CHECKING, Iterator, Dict, Text)
from typing_extensions import Self

from sortedcontainers import SortedList

from . import PYANNOTE_URI, PYANNOTE_SEGMENT, Timeline
from .base import BaseSegmentation, SegmentSetMixin
from .json import PYANNOTE_JSON, PYANNOTE_JSON_CONTENT
from .segment import Segment
from .utils.types import Support, Label, CropMode

if TYPE_CHECKING:
    from .annotation import Annotation


# =====================================================================
# Partition class
# =====================================================================

# TODO: Questions:
#  - "autofill" the partition if the initialized segments aren't filling?
#  - partition empty if only one segment?
#  - truthiness?

class Partition(SegmentSetMixin, BaseSegmentation):
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

    def __init__(self,
                 segments: Optional[Iterable[Segment]] = None,
                 start: float = 0.0,
                 end: float = None,
                 uri: str = None):
        segments = list(segments)
        if segments is None and end is None:
            raise ValueError("Cannot initialize an empty timeline without and end boundary")
        elif end is None:
            end = max(seg.end for seg in segments)
        elif not segments:
            segments = Segment(start, end)

        self.start = start
        self.end = end
        self._boundaries = None # TODO: figure out if needed
        super().__init__(uri)

        # TODO: check "filling"? autofill if not valid?
        self.update(self.gaps(support=self.extent()))
        if self[0].start < self.start or self[-1].end > self.end:
            raise ValueError(f"Segments have to be within ({start, end}) bounds")

    def __len__(self) -> int:
        pass

    def __nonzero__(self):
        return True

    def __bool__(self):
        return True

    def __eq__(self, other: Self):
        pass

    def __ne__(self, other: Self):
        pass

    def bisect(self, at: float):
        pass

    def add(self, segment: Segment):
        pass

    def remove(self, segment: Segment):
        pass

    def itersegments(self):
        pass

    def get_timeline(self) -> 'Timeline':
        pass

    def update(self, other: Self) -> Self:
        pass

    def co_iter(self, other: 'BaseSegmentation') -> Iterator[Tuple[Segment, Segment]]:
        pass

    def get_overlap(self) -> 'Timeline':
        pass

    def __str__(self):
        pass

    def __repr__(self):
        pass

    def __contains__(self, included: Union[Segment, 'Timeline']) -> bool:
        pass

    def empty(self) -> Self:
        pass

    def copy(self, segment_func: Optional[Callable[[Segment], Segment]] = None) -> Self:
        pass

    def extent(self) -> Segment:
        return Segment(self.start, self.end)

    def support_iter(self, collar: float = 0.0) -> Iterator[Segment]:
        pass

    def support(self, collar: float = 0.) -> 'Timeline':
        pass

    def crop_iter(self, support: Support, mode: CropMode = 'intersection', returns_mapping: bool = False) -> Iterator[
        Union[Tuple[Segment, Segment], Segment]]:
        pass

    def crop(self, support: Support, mode: CropMode = 'intersection', returns_mapping: bool = False) -> Union[
        Self, Tuple[Self, Dict[Segment, Segment]]]:
        pass

    def duration(self) -> float:
        pass

    def _repr_png_(self):
        pass
