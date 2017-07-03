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

.. ipython::

    In [1]: from pyannote.core import Annotation, Segment

    In [6]: annotation = Annotation()
       ...: annotation[Segment(1, 5)] = 'Carol'
       ...: annotation[Segment(6, 8)] = 'Bob'
       ...: annotation[Segment(12, 18)] = 'Carol'
       ...: annotation[Segment(7, 20)] = 'Alice'
       ...:

which is actually a shortcut for

.. ipython::

    In [6]: annotation = Annotation()
       ...: annotation[Segment(1, 5), '_'] = 'Carol'
       ...: annotation[Segment(6, 8), '_'] = 'Bob'
       ...: annotation[Segment(12, 18), '_'] = 'Carol'
       ...: annotation[Segment(7, 20), '_'] = 'Alice'
       ...:

where all tracks share the same (default) name ``'_'``.

In case two tracks share the same support, use a different track name:

.. ipython::

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

.. ipython::

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

from __future__ import unicode_literals

import six
import itertools
import operator
import warnings

import numpy as np

from . import PYANNOTE_URI, PYANNOTE_MODALITY, \
    PYANNOTE_SEGMENT, PYANNOTE_TRACK, PYANNOTE_LABEL
from xarray import DataArray
from sortedcontainers import SortedDict
from .segment import Segment
from .timeline import Timeline
from .json import PYANNOTE_JSON, PYANNOTE_JSON_CONTENT
from .util import string_generator, int_generator


class Annotation(object):
    """Annotation

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

    @classmethod
    def from_df(cls, df, uri=None, modality=None):

        df = df[[PYANNOTE_SEGMENT, PYANNOTE_TRACK, PYANNOTE_LABEL]]

        annotation = cls(uri=uri, modality=modality)

        for row in df.itertuples():
            if row[1] in annotation._tracks:
                annotation._tracks[row[1]][row[2]] = row[3]
            else:
                annotation._tracks[row[1]] = {row[2]: row[3]}

        annotation._labels = {label: None for label in df['label'].unique()}
        annotation._labelNeedsUpdate = {
            label: True for label in annotation._labels}

        annotation._timeline = None
        annotation._timelineNeedsUpdate = True

        return annotation

    def __init__(self, uri=None, modality=None):

        super(Annotation, self).__init__()

        self._uri = uri
        self.modality = modality

        # sorted dictionary
        # keys: annotated segments
        # values: {track: label} dictionary
        self._tracks = SortedDict()

        # dictionary
        # key: label
        # value: timeline
        self._labels = {}
        self._labelNeedsUpdate = {}

        # timeline meant to store all annotated segments
        self._timeline = None
        self._timelineNeedsUpdate = True

    def _get_uri(self):
        return self._uri

    def _set_uri(self, uri):
        # update uri for all internal timelines
        for label in self.labels():
            timeline = self.label_timeline(label, copy=False)
            timeline.uri = uri
        timeline = self.get_timeline(copy=False)
        timeline.uri = uri
        self._uri = uri

    uri = property(_get_uri, fset=_set_uri, doc="Resource identifier")

    def _updateLabels(self):

        # list of labels that needs to be updated
        update = set(
            label for label, update in self._labelNeedsUpdate.items() if update)

        # accumulate segments for updated labels
        _segments = {label: [] for label in update}
        for segment, track, label in self.itertracks(label=True):
            if label in update:
                _segments[label].append(segment)

        # create timeline with accumulated segments for updated labels
        for label in update:
            if _segments[label]:
                self._labels[label] = Timeline(
                    segments=_segments[label], uri=self.uri)
                self._labelNeedsUpdate[label] = False
            else:
                self._labels.pop(label, None)
                self._labelNeedsUpdate.pop(label, None)

    def __len__(self):
        """Number of segments

        >>> len(annotation)  # annotation contains three segments
        3
        """
        return len(self._tracks)

    def __nonzero__(self):
        return self.__bool__()

    def __bool__(self):
        """Emptiness

        >>> if annotation:
        ...    # annotation is empty
        ... else:
        ...    # annotation is not empty
        """
        return len(self._tracks) > 0

    def itersegments(self):
        """Iterate over segments (in chronological order)

        >>> for segment in annotation.itersegments():
        ...     # do something with the segment

        See also
        --------
        :class:`pyannote.core.Segment` describes how segments are sorted.
        """
        return iter(self._tracks)

    def itertracks(self, label=False, yield_label=False):
        """Iterate over tracks (in chronological order)

        Parameters
        ----------
        yield_label : bool, optional
            When True, yield (segment, track, label) tuples, such that
            annotation[segment, track] == label. Defaults to yielding
            (segment, track) tuple.

        Examples
        --------

        >>> for segment, track in annotation.itertracks():
        ...     # do something with the track

        >>> for segment, track, label in annotation.itertracks(yield_label=True):
        ...     # do something with the track and its label
        """

        if label:
            warnings.warn(
                '"label" parameter has been renamed to "yield_label".',
                DeprecationWarning
            )
            yield_label = label

        for segment, tracks in self._tracks.items():
            for track, lbl in sorted(
                six.iteritems(tracks),
                key=lambda tl: (str(tl[0]), str(tl[1]))):
                if yield_label:
                    yield segment, track, lbl
                else:
                    yield segment, track

    def _updateTimeline(self):
        self._timeline = Timeline(segments=self._tracks, uri=self.uri)
        self._timelineNeedsUpdate = False

    def get_timeline(self, copy=True):
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
            self._updateTimeline()
        if copy:
            return self._timeline.copy()
        return self._timeline

    def __eq__(self, other):
        """Equality

        >>> annotation == other

        Two annotations are equal if and only if their tracks and associated
        labels are equal.
        """
        pairOfTracks = six.moves.zip_longest(self.itertracks(label=True),
                                             other.itertracks(label=True))
        return all(t1 == t2 for t1, t2 in pairOfTracks)

    def __ne__(self, other):
        """Inequality"""
        pairOfTracks = six.moves.zip_longest(self.itertracks(label=True),
                                             other.itertracks(label=True))

        return any(t1 != t2 for t1, t2 in pairOfTracks)

    def __contains__(self, included):
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

    def crop(self, support, mode='intersection'):
        """Crop annotation to new support

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
        cropped : Annotation
            Cropped annotation

        Note
        ----
        In 'intersection' mode, the best is done to keep the track names
        unchanged. However, in some cases where two original segments are
        cropped into the same resulting segments, conflicting track names are
        modified to make sure no track is lost.

        """

        # TODO speed things up by working directly with annotation internals

        if isinstance(support, Segment):
            support = Timeline(segments=[support], uri=self.uri)
            return self.crop(support, mode=mode)

        elif isinstance(support, Timeline):

            cropped = self.__class__(uri=self.uri, modality=self.modality)

            if mode == 'loose':

                _tracks = {}
                _labels = set([])

                for segment, _ in \
                    self.get_timeline(copy=False).co_iter(support):

                    tracks = dict(self._tracks[segment])
                    _tracks[segment] = tracks
                    _labels.update(tracks.values())

                cropped._tracks = SortedDict(_tracks)

                cropped._labelNeedsUpdate = {label: True for label in _labels}
                cropped._labels = {label: None for label in _labels}

                cropped._timelineNeedsUpdate = True
                cropped._timeline = None

                return cropped

            elif mode == 'strict':

                _tracks = {}
                _labels = set([])

                for segment, other_segment in \
                        self.get_timeline(copy=False).co_iter(support):

                    if segment not in other_segment:
                        continue

                    tracks = dict(self._tracks[segment])
                    _tracks[segment] = tracks
                    _labels.update(tracks.values())

                cropped._tracks = SortedDict(_tracks)

                cropped._labelNeedsUpdate = {label: True for label in _labels}
                cropped._labels = {label: None for label in _labels}

                cropped._timelineNeedsUpdate = True
                cropped._timeline = None

                return cropped

            elif mode == 'intersection':

                for segment, other_segment in \
                        self.get_timeline(copy=False).co_iter(support):

                    intersection = segment & other_segment
                    for track, label in six.iteritems(self._tracks[segment]):
                        track = cropped.new_track(intersection,
                                                  candidate=track)
                        cropped[intersection, track] = label

                return cropped

            else:
                raise NotImplementedError("unsupported mode: '%s'" % mode)


    def get_tracks(self, segment):
        """Query tracks by segment

        Parameters
        ----------
        segment : Segment
            Query

        Returns
        -------
        tracks : set
            Set of tracks

        Note
        ----
        This will return an empty set if segment does not exist.
        """
        return set(self._tracks.get(segment, {}))

    def has_track(self, segment, track):
        """Check whether a given track exists

        Parameters
        ----------
        segment : Segment
            Query segment
        track :
            Query track

        Returns
        -------
        exists : bool
            True if track exists for segment
        """
        return track in self._tracks.get(segment, {})

    def copy(self):
        """Get a copy of the annotation

        Returns
        -------
        annotation : Annotation
            Copy of the annotation
        """

        # create new empty annotation
        copied = self.__class__(uri=self.uri, modality=self.modality)

        # deep copy internal track dictionary
        _tracks, _labels = [], set([])
        for key, value in self._tracks.items():
            _labels.update(value.values())
            _tracks.append((key, dict(value)))

        copied._tracks = SortedDict(_tracks)

        copied._labels = {label: None for label in _labels}
        copied._labelNeedsUpdate = {label: True for label in _labels}

        copied._timeline = None
        copied._timelineNeedsUpdate = True

        return copied

    def new_track(self, segment, candidate=None, prefix=None):
        """Generate a new track name for given segment

        Ensures that the returned track name does not already
        exist for the given segment.

        Parameters
        ----------
        segment : Segment
            Segment for which a new track name is generated.
        candidate : any valid track name, optional
            When provided, try this candidate name first.
        prefix : str, optional
            Track name prefix. Defaults to the empty string ''.

        Returns
        -------
        name : str
            New track name
        """

        # obtain list of existing tracks for segment
        existing_tracks = set(self._tracks.get(segment, {}))

        # if candidate is provided, check whether it already exists
        # in case it does not, use it
        if (candidate is not None) and (candidate not in existing_tracks):
            return candidate

        # no candidate was provided or the provided candidate already exists
        # we need to create a brand new one

        # by default (if prefix is not provided), use ''
        if prefix is None:
            prefix = ''

        # find first non-existing track name for segment
        # eg. if '0' exists, try '1', then '2', ...
        count = 0
        while ('%s%d' % (prefix, count)) in existing_tracks:
            count += 1

        # return first non-existing track name
        return '%s%d' % (prefix, count)

    def __str__(self):
        """Human-friendly representation"""
        # TODO: use pandas.DataFrame
        return "\n".join(["%s %s %s" % (s, t, l)
                          for s, t, l in self.itertracks(label=True)])

    def __delitem__(self, key):
        """Delete one track

        >>> del annotation[segment, track]

        Delete all tracks of a segment

        >>> del annotation[segment]
        """

        # del annotation[segment]
        if isinstance(key, Segment):

            # Pop segment out of dictionary
            # and get corresponding tracks
            # Raises KeyError if segment does not exist
            tracks = self._tracks.pop(key)

            # mark timeline as modified
            self._timelineNeedsUpdate = True

            # mark every label in tracks as modified
            for track, label in six.iteritems(tracks):
                self._labelNeedsUpdate[label] = True

        # del annotation[segment, track]
        elif isinstance(key, tuple) and len(key) == 2:

            # get segment tracks as dictionary
            # if segment does not exist, get empty dictionary
            # Raises KeyError if segment does not exist
            tracks = self._tracks[key[0]]

            # pop track out of tracks dictionary
            # and get corresponding label
            # Raises KeyError if track does not exist
            label = tracks.pop(key[1])

            # mark label as modified
            self._labelNeedsUpdate[label] = True

            # if tracks dictionary is now empty,
            # remove segment as well
            if not tracks:
                self._tracks.pop(key[0])
                self._timelineNeedsUpdate = True

        else:
            raise NotImplementedError(
                'Deletion only works with Segment or (Segment, track) keys.')

    # label = annotation[segment, track]
    def __getitem__(self, key):
        """Get track label

        >>> label = annotation[segment, track]

        Note
        ----
        ``annotation[segment]`` is equivalent to ``annotation[segment, '_']``

        """

        if isinstance(key, Segment):
            key = (key, '_')

        return self._tracks[key[0]][key[1]]

    # annotation[segment, track] = label
    def __setitem__(self, key, label):
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
        if segment not in self._tracks:
            self._tracks[segment] = {}
            self._timelineNeedsUpdate = True

        # in case we modify an existing track
        # mark old label as modified
        if track in self._tracks[segment]:
            old_label = self._tracks[segment][track]
            self._labelNeedsUpdate[old_label] = True

        # mark new label as modified
        self._tracks[segment][track] = label
        self._labelNeedsUpdate[label] = True

    def empty(self):
        """Return an empty copy

        Returns
        -------
        empty : Annotation
            Empty annotation using the same 'uri' and 'modality' attributes.

        """
        return self.__class__(uri=self.uri, modality=self.modality)

    def labels(self):
        """Get sorted list of labels

        Returns
        -------
        labels : list
            Sorted list of labels
        """
        if any([lnu for lnu in self._labelNeedsUpdate.values()]):
            self._updateLabels()
        return sorted(self._labels, key=str)

    def get_labels(self, segment, unique=True):
        """Query labels by segment

        Parameters
        ----------
        segment : Segment
            Query
        unique : bool, optional
            When False, return the list of (possibly repeated) labels.
            Defaults to returning the set of labels.

        Returns
        -------
        labels : set
            Set of labels for `segment` if it exists, empty set otherwise.

        Examples
        --------
        >>> annotation = Annotation()
        >>> segment = Segment(0, 2)
        >>> annotation[segment, 'speaker1'] = 'Bernard'
        >>> annotation[segment, 'speaker2'] = 'John'
        >>> print sorted(annotation.get_labels(segment))
        set(['Bernard', 'John'])
        >>> print annotation.get_labels(Segment(1, 2))
        set([])

        """

        labels = self._tracks.get(segment, {}).values()

        if unique:
            return set(labels)

        return labels

    def subset(self, labels, invert=False):
        """Filter annotation by labels

        Parameters
        ----------
        labels : iterable
            List of filtered labels
        invert : bool, optional
            If invert is True, extract all but requested labels

        Returns
        -------
        filtered : Annotation
            Filtered annotation
        """

        labels = set(labels)

        if invert:
            labels = set(self.labels()) - labels
        else:
            labels = labels & set(self.labels())

        sub = self.__class__(uri=self.uri, modality=self.modality)

        _tracks, _labels = {}, set([])
        for segment, tracks in self._tracks.items():
            sub_tracks = {track: label for track, label in tracks.items()
                                       if label in labels}
            if sub_tracks:
                _tracks[segment] = sub_tracks
                _labels.update(sub_tracks.values())

        sub._tracks = SortedDict(_tracks)

        sub._labelNeedsUpdate = {label: True for label in _labels}
        sub._labels = {label: None for label in _labels}

        sub._timelineNeedsUpdate = True
        sub._timeline = None

        return sub

    def update(self, annotation, copy=False):
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
        for segment, track, label in annotation.itertracks(label=True):
            result[segment, track] = label

        return result

    def label_timeline(self, label, copy=True):
        """Query segments by label

        Parameters
        ----------
        label : object
            Query
        copy : bool, optional
            Defaults (True) to returning a copy of the internal timeline.
            Set to False to return the actual internal timeline (faster).

        Returns
        -------
        timeline : Timeline
            Timeline made of all segments for which at least one track is
            annotated as label

        Note
        ----
        If label does not exist, this will return an empty timeline.

        Note
        ----
        In case copy is set to False, be careful **not** to modify the returned
        timeline, as it may lead to weird subsequent behavior of the annotation
        instance.

        """
        if label not in self.labels():
            return Timeline(uri=self.uri)

        if self._labelNeedsUpdate[label]:
            self._updateLabels()

        if copy:
            return self._labels[label].copy()

        return self._labels[label]

    def label_coverage(self, label):
        warnings.warn(
            '"label_coverage" has been renamed to "label_support".',
            DeprecationWarning)
        return self.label_support(label)

    def label_support(self, label):
        """Label support

        Equivalent to ``Annotation.label_timeline(label).support()``

        Parameters
        ----------
        label : object
            Query

        Returns
        -------
        support : Timeline
            Label support

        See also
        --------
        :func:`~pyannote.core.Annotation.label_timeline`
        :func:`~pyannote.core.Timeline.support`

        """
        return self.label_timeline(label, copy=False).support()

    def label_duration(self, label):
        """Label duration

        Equivalent to ``Annotation.label_timeline(label).duration()``

        Parameters
        ----------
        label : object
            Query

        Returns
        -------
        duration : float
            Duration, in seconds.

        See also
        --------
        :func:`~pyannote.core.Annotation.label_timeline`
        :func:`~pyannote.core.Timeline.duration`

        """

        return self.label_timeline(label, copy=False).duration()

    def chart(self, percent=False):
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

    def argmax(self, support=None, segment=None):
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

        if segment is not None:
            warnings.warn(
                '"segment" parameter has been renamed to "support".',
                DeprecationWarning)
            support = segment

        cropped = self
        if support is not None:
            cropped = cropped.crop(support, mode='intersection')

        if not cropped:
            return None

        return max(((_, cropped.label_duration(_)) for _ in cropped.labels()),
                   key=lambda x: x[1])[0]

    def translate(self, translation):
        warnings.warn(
            '"translate" has been replaced by "rename_labels".',
            DeprecationWarning)
        return self.rename_labels(mapping=translation)

    def __mod__(self, translation):
        warnings.warn(
            'support for "%" operator will be removed.',
            DeprecationWarning)
        return self.rename_labels(mapping=translation)

    def retrack(self):
        warnings.warn(
            '"retrack" has been renamed to "rename_tracks".',
            DeprecationWarning)
        return self.rename_tracks(generator='int')

    def rename_tracks(self, generator='string'):
        """Rename all tracks

        Parameters
        ----------
        generator : 'string', 'int', or iterable, optional
            If 'string' (default) rename tracks to 'A', 'B', 'C', etc.
            If 'int', rename tracks to 0, 1, 2, etc.
            If iterable, use it to generate track names.

        Returns
        -------
        renamed : Annotation
            Copy of the original annotation where tracks are renamed.

        Example
        -------
        >>> annotation = Annotation()
        >>> annotation[Segment(0, 1), 'a'] = 'a'
        >>> annotation[Segment(0, 1), 'b'] = 'b'
        >>> annotation[Segment(1, 2), 'a'] = 'a'
        >>> annotation[Segment(1, 3), 'c'] = 'c'
        >>> print(annotation)
        [ 00:00:00.000 -->  00:00:01.000] a a
        [ 00:00:00.000 -->  00:00:01.000] b b
        [ 00:00:01.000 -->  00:00:02.000] a a
        [ 00:00:01.000 -->  00:00:03.000] c c
        >>> print(annotation.rename_tracks(generator='int'))
        [ 00:00:00.000 -->  00:00:01.000] 0 a
        [ 00:00:00.000 -->  00:00:01.000] 1 b
        [ 00:00:01.000 -->  00:00:02.000] 2 a
        [ 00:00:01.000 -->  00:00:03.000] 3 c
        """

        renamed = self.__class__(uri=self.uri, modality=self.modality)

        if generator == 'string':
            generator = string_generator()
        elif generator == 'int':
            generator = int_generator()

        # TODO speed things up by working directly with annotation internals
        for s, _, label in self.itertracks(label=True):
            renamed[s, next(generator)] = label
        return renamed

    def rename_labels(self, mapping=None, generator='string', copy=True):
        """Rename labels

        Parameters
        ----------
        mapping : dict, optional
            {old_name: new_name} mapping dictionary.
        generator : 'string', 'int' or iterable, optional
            If 'string' (default) rename label to 'A', 'B', 'C', ... If 'int',
            rename to 0, 1, 2, etc. If iterable, use it to generate labels.
        copy : bool, optional
            Return a copy of the annotation. Defaults to updating the
            annotation in-place.

        Returns
        -------
        renamed : Annotation
            Annotation where labels have been renamed

        Note
        ----
        Unmapped labels are kept unchanged.

        Note
        ----
        Parameter `generator` has no effect when `mapping` is provided.

        """

        if mapping is None:
            if generator == 'string':
                generator = string_generator()
            elif generator == 'int':
                generator = int_generator()
            # generate mapping
            mapping = {label: next(generator) for label in self.labels()}

        renamed = self.copy() if copy else self

        for old_label, new_label in mapping.items():
            renamed._labelNeedsUpdate[old_label] = True
            renamed._labelNeedsUpdate[new_label] = True

        for segment, tracks in self._tracks.items():
            new_tracks = {track: mapping.get(label, label)
                          for track, label in tracks.items()}
            renamed._tracks[segment] = new_tracks

        return renamed

    def anonymize_labels(self, generator='string'):
        warnings.warn(
            "'anonymize_labels' has been replaced by 'rename_labels'",
            DeprecationWarning)
        return self.rename_labels(generator=generator)

    def relabel_tracks(self, generator='string'):
        """Relabel tracks

        Create a new annotation where each track has a unique label.

        Parameters
        ----------
        generator : 'string', 'int' or iterable, optional
            If 'string' (default) relabel tracks to 'A', 'B', 'C', ... If 'int'
            relabel to 0, 1, 2, ... If iterable, use it to generate labels.

        Returns
        -------
        renamed : Annotation
            New annotation with relabeled tracks.
        """

        if generator == 'string':
            generator = string_generator()
        elif generator == 'int':
            generator = int_generator()

        relabeled = self.empty()
        for s, t, _ in self.itertracks(label=True):
            relabeled[s, t] = next(generator)

        return relabeled

    def anonymize_tracks(self, generator='string'):
        warnings.warn(
            "'anonymize_tracks' has been replaced by 'relabel_tracks'",
            DeprecationWarning)
        return self.relabel_tracks(generator=generator)

    def support(self, collar=0.):
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
            if collar > 0.:
                gaps = timeline.gaps()
                for gap in gaps:
                    if gap.duration < collar:
                        timeline.add(gap)

            # reconstruct annotation with merged tracks
            for segment in timeline.support():
                support[segment, next(generator)] = label

        return support

    def smooth(self, collar=0.):
        warnings.warn(
            '"smooth" has been renamed to "support".',
            DeprecationWarning)
        return self.support(collar=collar)

    def co_iter(self, other):
        """Iterate over pairs of intersecting tracks

        Parameters
        ----------
        other : Annotation
            Second annotation

        Returns
        -------
        iterable : (Segment, object), (Segment, object) iterable
            Yields pairs of intersectins tracks, in chronological (then
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

    def __mul__(self, other):
        """Cooccurrence (or confusion) matrix

        >>> matrix = annotation * other
        >>> matrix.loc['A', 'a']   # duration of cooccurrence between labels
                                   # 'A' from `annotation` and 'a' from `other`

        Parameters
        ----------
        other : Annotation
            Second annotation

        Returns
        -------
        cooccurrence : DataArray
        """

        if not isinstance(other, Annotation):
            raise TypeError(
                'computing cooccurrence matrix only works with Annotation '
                'instances.')

        # initialize matrix with one row per label in `self`,
        # and one column per label in `other`

        i = self.labels()
        j = other.labels()
        matrix = DataArray(
            np.zeros((len(i), len(j))),
            coords=[('i', i), ('j', j)])

        # iterate over intersecting tracks and accumulate durations
        for (segment, track), (other_segment, other_track) in self.co_iter(other):
            label = self[segment, track]
            other_label = other[other_segment, other_track]
            duration = (segment & other_segment).duration
            matrix.loc[label, other_label] += duration

        return matrix

    def for_json(self):
        """Serialization

        See also
        --------
        :mod:`pyannote.core.json`
        """

        data = {PYANNOTE_JSON: self.__class__.__name__}
        content = [{PYANNOTE_SEGMENT: s.for_json(),
                    PYANNOTE_TRACK: t,
                    PYANNOTE_LABEL: l}
                   for s, t, l in self.itertracks(label=True)]
        data[PYANNOTE_JSON_CONTENT] = content

        if self.uri:
            data[PYANNOTE_URI] = self.uri

        if self.modality:
            data[PYANNOTE_MODALITY] = self.modality

        return data

    @classmethod
    def from_json(cls, data):
        """Deserialization

        See also
        --------
        :mod:`pyannote.core.json`
        """

        uri = data.get(PYANNOTE_URI, None)
        modality = data.get(PYANNOTE_MODALITY, None)
        annotation = cls(uri=uri, modality=modality)
        for one in data[PYANNOTE_JSON_CONTENT]:
            segment = Segment.from_json(one[PYANNOTE_SEGMENT])
            track = one[PYANNOTE_TRACK]
            label = one[PYANNOTE_LABEL]
            annotation[segment, track] = label

        return annotation

    def _repr_png_(self):
        """IPython notebook support

        See also
        --------
        :mod:`pyannote.core.notebook`
        """

        from .notebook import repr_annotation
        return repr_annotation(self)
