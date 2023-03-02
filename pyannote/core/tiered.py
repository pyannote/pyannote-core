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
from pathlib import Path
from typing import Optional, Dict, Union, Iterable, List, TextIO, Tuple, Iterator, Callable, Type, Generic

from sortedcontainers import SortedDict
from typing_extensions import Self

from pyannote.core import Annotation
from .base import BaseSegmentation, GappedAnnotationMixin, ContiguousAnnotationMixin
from .partition import Partition
from .segment import Segment
from .timeline import Timeline
from .utils.generators import string_generator
from .utils.types import Label, Key, Support, TierName, CropMode, ContiguousSupport, TierItemPair

# TODO: add JSON dumping/loading
# TODO: QUESTIONS:
#  - iterator for the TieredAnnotation

# TODO: add segmentation abstract class

# TODO: IDEA: use a timeline in the Tier to do all the cropping/etc/ operations
#  and just make this class a thin wrapper for it
T = Type[Union[Partition, Timeline]]


class BaseTier(BaseSegmentation, Generic[T]):
    _segmentation_type: T

    # TODO: handle segment sets changes for
    #  - add (partition)
    #  - bisect (partition)
    #  - crop (partition/timeline)
    #  - for extrusion, should be based on cropping

    def __init__(self, name: str = None, uri: str = None):
        super().__init__(uri)
        self.name = name

        self._segmentation = self._segmentation_type()
        self._segments: Dict[Segment, TierLabel] = dict()

    def __setitem__(self, segment: Segment, label: str):
        # TODO: check
        self._segmentation.add(segment)
        self._segments[segment] = label

    def __getitem__(self, key: Union[Segment, int]) -> str:
        # TODO: check
        if isinstance(key, int):
            key = self._segmentation.__getitem__(key)
        return self._segments[key]

    def __delitem__(self, key: Union[Segment, int]):
        # TODO: check
        if isinstance(key, int):
            key = self._segmentation.__getitem__(key)

        del self._segments[key]
        self._segmentation.remove(key)

    def __iter__(self) -> Iterable[TierItemPair]:
        """Return segments with their annotation, in chronological order"""
        for segment in self._segmentation.itersegments():
            yield segment, self._segments[segment]

    def __len__(self):
        """Number of segments in the tier

        >>> len(tier)  # tier contains three segments
        3
        """
        return len(self._segments)

    def __bool__(self):
        """Emptiness

        >>> if tier:
        ...    # tier is empty
        ... else:
        ...    # tier is not empty
        """
        return bool(self._segments)

    def __eq__(self, other: 'BaseTier'):
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

    def __ne__(self, other: 'BaseTier'):
        """Inequality"""
        return self._segments != other._segments

    def itersegments(self):
        return self._segmentation.itersegments()

    def empty(self) -> 'BaseTier':
        """Return an empty copy

        Returns
        -------
        empty : Tier
            Empty timeline using the same 'uri' attribute.

        """
        return self.__class__(self.name, uri=self.uri)

    def update(self, tier: 'BaseTier') -> 'BaseTier':
        pass  # TODO

    def copy(self, segment_func: Optional[Callable[[Segment], Segment]] = None) -> Self:
        pass  # TODO

    def extent(self) -> Segment:
        return self._segmentation.extent()

    def crop_iter(self, support: Support, mode: CropMode = 'intersection', returns_mapping: bool = False) -> Iterator[
        Union[Tuple[Segment, Segment], Segment]]:
        pass

    def duration(self) -> float:
        return self._segmentation.duration()

    def _repr_png_(self):
        pass


class Tier(GappedAnnotationMixin, BaseTier[Timeline]):
    _segmentation_type = Timeline
    """A set of chronologically-ordered, optionally non-overlapping
    and annotated segments"""

    def gaps_iter(self, support: Optional[Support] = None) -> Iterator[Segment]:
        return self._segmentation.gaps_iter(support)

    def gaps(self, support: Optional[Support] = None) -> 'Timeline':
        return self._segmentation.gaps(support)

    def extrude(self, removed: Support, mode: CropMode = 'intersection') -> Self:
        return self._segmentation.extrude(removed, mode)

    def crop(self, support: Support, mode: CropMode = 'intersection', returns_mapping: bool = False) -> Union[
        Self, Tuple[Self, Dict[Segment, Segment]]]:
        # TODO (for segments mapping):
        #  - if loose/strict, use segment_set of cropped segmentation to find deleted segments
        #  - if intersection, use return_mapping to replace sliced segments
        return self._segmentation.crop(support, mode, returns_mapping)

    def support(self, collar: float = 0.) -> Timeline:
        return self._segmentation.support(collar)

    def get_overlap(self) -> 'Timeline':
        return self._segmentation.get_overlap()

    def co_iter(self, other: Union[Timeline, Segment]) -> Iterator[Tuple[Segment, Segment]]:
        # TODO : Doc
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

        yield from self._segmentation.co_iter(other)


class PartitionTier(ContiguousAnnotationMixin, BaseTier[Partition]):
    _segmentation_type = Partition
    # TODO:
    # - __iter__ should also yield empty segments, with a None annotation

    def crop(self,
             support: ContiguousSupport,
             mode: CropMode = 'intersection',
             returns_mapping: bool = False) -> Union[Self, Tuple[Self, Dict[Segment, Segment]]]:
        # TODO:
        #  - think about using crop_iter first
        pass


class TieredAnnotation(GappedAnnotationMixin, BaseSegmentation):

    def __init__(self, uri: Optional[str] = None):
        super().__init__(uri)

        self._uri: Optional[str] = uri

        # sorted dictionary
        # values: {tiername: tier} dictionary
        self._tiers: Dict[TierName, Tier] = SortedDict()

        # timeline meant to store all annotated segments
        self._timeline: Timeline = None
        self._timelineNeedsUpdate: bool = True

    @classmethod
    def from_textgrid(cls, textgrid: Union[str, Path, TextIO],
                      textgrid_format: str = "full"):
        try:
            from textgrid_parser import parse_textgrid
        except ImportError:
            raise ImportError("The dependencies used to parse TextGrid file cannot be found. "
                              "Please install using pyannote.core[textgrid]")
        # TODO : check for tiers with duplicate names

        return parse_textgrid(textgrid, textgrid_format=textgrid_format)

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
    def tiers(self) -> List[Tier]:
        return list(self._tiers.values())

    @property
    def tiers_names(self) -> List[TierName]:
        return list(self._tiers.keys())

    @property
    def tiers_count(self):
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
        # TODO : docfix
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
        for tier in self.tiers:
            yield from tier

    def __iter__(self) -> Iterable[Tuple[Segment, str]]:
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

    def __eq__(self, other: 'TieredAnnotation'):
        """Equality

        >>> annotation == other

        Two annotations are equal if and only if their tracks and associated
        labels are equal.
        """
        # TODO
        pairOfTracks = itertools.zip_longest(
            self.itertracks(yield_label=True),
            other.itertracks(yield_label=True))
        return all(t1 == t2 for t1, t2 in pairOfTracks)

    def __ne__(self, other: 'TieredAnnotation'):
        """Inequality"""
        # TODO
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
            -> 'TieredAnnotation':
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
        cropped : TieredAnnotation
            Cropped textgrid
        """
        new_tg = TieredAnnotation(self.uri)
        for tier_name, tier in self._tiers.items():
            new_tg._tiers[tier_name] = tier.crop(support)
        return new_tg

    def copy(self) -> 'TieredAnnotation':
        """Get a copy of the annotation

        Returns
        -------
        annotation : TieredAnnotation
            Copy of the textgrid
        """

        # create new empty annotation
        # TODO
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

    def __getitem__(self, key: TierName) -> Tier:
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

    def update(self, textgrid: 'TieredAnnotation', copy: bool = False) \
            -> 'TieredAnnotation':
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

        # TODO

        return result

    def support(self, collar: float = 0.) -> 'TieredAnnotation':
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

    def _repr_png(self):
        """IPython notebook support

        See also
        --------
        :mod:`pyannote.core.notebook`
        """

        from .notebook import repr_annotation
        return repr_annotation(self)
