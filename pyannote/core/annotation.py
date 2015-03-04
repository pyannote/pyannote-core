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

import itertools
import operator
import warnings

import numpy as np

from . import PYANNOTE_URI, PYANNOTE_MODALITY, \
    PYANNOTE_SEGMENT, PYANNOTE_TRACK, PYANNOTE_LABEL
from banyan import SortedDict
from interval_tree import TimelineUpdator
from segment import Segment
from timeline import Timeline
from json import PYANNOTE_JSON, PYANNOTE_JSON_CONTENT

# ignore Banyan warning
warnings.filterwarnings(
    'ignore', 'Key-type optimization',
    Warning, 'pyannote.core.annotation'
)


class Unknown(object):

    nextID = 0

    @classmethod
    def reset(cls):
        cls.nextID = 0

    @classmethod
    def next(cls):
        cls.nextID += 1
        return cls.nextID

    def __init__(self, format='#{id:d}'):
        super(Unknown, self).__init__()
        self.ID = Unknown.next()
        self._format = format

    def __str__(self):
        return self._format.format(id=self.ID)

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash(self.ID)

    def __eq__(self, other):
        if isinstance(other, Unknown):
            return self.ID == other.ID
        else:
            return False


class Annotation(object):
    """Annotation

    Parameters
    ----------
    uri : string, optional
        uniform resource identifier of annotated document
    modality : string, optional
        name of annotated modality

    """

    @classmethod
    def from_df(cls, df, uri=None, modality=None):
        """

        Parameters
        ----------
        df : DataFrame
            Must contain the following columns: 'segment', 'track' and 'label'
        uri : str, optional
            Resource identifier
        modality : str, optional
            Modality

        Returns
        -------

        """
        annotation = cls(uri=uri, modality=modality)
        for _, (segment, track, label) in df[
            [PYANNOTE_SEGMENT,
             PYANNOTE_TRACK,
             PYANNOTE_LABEL]
        ].iterrows():
            annotation[segment, track] = label

        return annotation

    def __init__(self, uri=None, modality=None):
        super(Annotation, self).__init__()

        self._uri = uri
        self.modality = modality

        # sorted dictionary
        # keys: annotated segments
        # values: {track: label} dictionary
        self._tracks = SortedDict(key_type=(float, float),
                                  updator=TimelineUpdator)

        # dictionary
        # key: label
        # value: timeline
        self._labels = {}
        self._labelNeedsUpdate = {}

        # timeline meant to store all annotated segments
        self._timeline = Timeline(uri=uri)
        self._timelineNeedsUpdate = True

    def _get_uri(self):
        return self._uri

    def _set_uri(self, uri):
        # update uri for all internal timelines
        for _, timeline in self._labels.iteritems():
            timeline.uri = uri
        self._uri = uri

    uri = property(_get_uri, fset=_set_uri, doc="Resource identifier")

    def _updateLabels(self):

        # (re-)initialize changed label timeline
        for l, needsUpdate in self._labelNeedsUpdate.iteritems():
            if needsUpdate:
                self._labels[l] = Timeline(uri=self.uri)

        # fill changed label timeline
        for segment, track, l in self.itertracks(label=True):
            if self._labelNeedsUpdate[l]:
                self._labels[l].add(segment)

        self._labelNeedsUpdate = {l: False for l in self._labels}

        # remove "ghost" labels (i.e. label with empty timeline)
        labels = self._labels.keys()
        for l in labels:
            if not self._labels[l]:
                self._labels.pop(l)
                self._labelNeedsUpdate.pop(l)

    def __len__(self):
        """Number of segments"""
        return self._tracks.length()

    def __nonzero__(self):
        return self._tracks.length() > 0

    def itersegments(self):
        """Segment iterator"""
        return iter(self._tracks)

    def itertracks(self, label=False):
        for segment, tracks in self._tracks.items():
            for track, lbl in tracks.iteritems():
                if label:
                    yield segment, track, lbl
                else:
                    yield segment, track

    def _updateTimeline(self):
        self._timeline = Timeline(segments=self._tracks, uri=self.uri)
        self._timelineNeedsUpdate = False

    def get_timeline(self):
        """Get timeline made of annotated segments"""
        if self._timelineNeedsUpdate:
            self._updateTimeline()
        return self._timeline

    def __eq__(self, other):
        return self._tracks == other._tracks

    def __ne__(self, other):
        return self._tracks != other._tracks

    def __contains__(self, included):
        """Inclusion

        Use expression 'segment in annotation' or 'timeline in annotation'

        Parameters
        ----------
        included : `Segment` or `Timeline`

        Returns
        -------
        contains : bool
            True if every segment in `included` exists in annotation
            False otherwise

        """
        return included in self.get_timeline()

    def crop(self, other, mode='intersection'):
        """Crop annotation

        Parameters
        ----------
        other : `Segment` or `Timeline`

        mode : {'strict', 'loose', 'intersection'}
            In 'strict' mode, only segments fully included in focus coverage
            are kept. In 'loose' mode, any intersecting segment is kept
            unchanged. In 'intersection' mode, only intersecting segments are
            kept and replaced by their actual intersection with the focus.

        Returns
        -------
        cropped : Annotation

        Remarks
        -------
        In 'intersection' mode, the best is done to keep the track names
        unchanged. However, in some cases where two original segments are
        cropped into the same resulting segments, conflicting track names are
        modified to make sure no track is lost.
        """

        if isinstance(other, Segment):
            other = Timeline(segments=[other], uri=self.uri)
            cropped = self.crop(other, mode=mode)

        elif isinstance(other, Timeline):

            cropped = self.__class__(uri=self.uri, modality=self.modality)

            if mode == 'loose':
                # TODO
                # update co_iter to yield (segment, tracks), (segment, tracks)
                # instead of segment, segment
                # This would avoid calling ._tracks.get(segment)
                for segment, _ in self.get_timeline().co_iter(other):
                    for track, label in self._tracks[segment].iteritems():
                        cropped[segment, track] = label

            elif mode == 'strict':
                # TODO
                # see above
                for segment, other_segment in \
                        self.get_timeline().co_iter(other):

                    if segment in other_segment:
                        for track, label in self._tracks[segment].iteritems():
                            cropped[segment, track] = label

            elif mode == 'intersection':
                # see above
                for segment, other_segment in \
                        self.get_timeline().co_iter(other):

                    intersection = segment & other_segment
                    for track, label in self._tracks[segment].iteritems():
                        track = cropped.new_track(intersection,
                                                  candidate=track)
                        cropped[intersection, track] = label

            else:
                raise NotImplementedError("unsupported mode: '%s'" % mode)

        return cropped

    def get_tracks(self, segment):
        """Set of tracks for query segment

        Parameters
        ----------
        segment : `Segment`
            Query segment

        Returns
        -------
        tracks : set
            Set of tracks for query segment
        """
        return set(self._tracks.get(segment, {}))

    def has_track(self, segment, track):
        """Check whether a given track exists

        Parameters
        ----------
        segment : `Segment`
            Query segment
        track :
            Query track

        Returns
        -------
        exists : bool
            True if track exists for segment
        """
        return track in self._tracks.get(segment, {})

    def get_track_by_name(self, track):
        """Get all tracks with given name

        Parameters
        ----------
        track : any valid track name
            Requested name track

        Returns
        -------
        tracks : list
            List of (segment, track) tuples
        """
        raise NotImplementedError('')

    def copy(self):

        # create new empty annotation
        copied = self.__class__(uri=self.uri, modality=self.modality)

        # deep copy internal track dictionary
        _tracks = [(key, dict(value)) for (key, value) in self._tracks.items()]
        copied._tracks = SortedDict(items=_tracks,
                                    key_type=(float, float),
                                    updator=TimelineUpdator)

        # deep copy internal label timelines
        _labels = {key: timeline.copy()
                   for (key, timeline) in self._labels.iteritems()}
        copied._labels = _labels

        # deep copy need-update indicator
        copied._labelNeedsUpdate = dict(self._labelNeedsUpdate)

        copied._timelineNeedsUpdate = self._timelineNeedsUpdate

        return copied

    def retrack(self):
        """
        """
        retracked = self.__class__(uri=self.uri, modality=self.modality)
        for n, (s, _, label) in enumerate(self.itertracks(label=True)):
            retracked[s, n] = label
        return retracked

    def new_track(self, segment, candidate=None, prefix=None):
        """Track name generator

        Parameters
        ----------
        segment : Segment
        prefix : str, optional
        candidate : any valid track name

        Returns
        -------
        track : str
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

        # del annotation[segment]
        if isinstance(key, Segment):

            # Pop segment out of dictionary
            # and get corresponding tracks
            # Raises KeyError if segment does not exist
            tracks = self._tracks.pop(key)

            # mark timeline as modified
            self._timelineNeedsUpdate = True

            # mark every label in tracks as modified
            for track, label in tracks.iteritems():
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
            raise KeyError('')

    # label = annotation[segment, track]
    def __getitem__(self, key):

        if isinstance(key, Segment):
            key = (key, '_')

        return self._tracks[key[0]][key[1]]

    # annotation[segment, track] = label
    def __setitem__(self, key, label):

        if isinstance(key, Segment):
            key = (key, '_')

        # in case we create a new segment
        # mark timeline as modified
        if key[0] not in self._tracks:
            self._tracks[key[0]] = {}
            self._timelineNeedsUpdate = True

        # in case we modify an existing track
        # mark old label as modified
        if key[1] in self._tracks[key[0]]:
            old_label = self._tracks[key[0]][key[1]]
            self._labelNeedsUpdate[old_label] = True

        # mark new label as modified
        self._tracks[key[0]][key[1]] = label
        self._labelNeedsUpdate[label] = True

    def empty(self):
        return self.__class__(uri=self.uri, modality=self.modality)

    @staticmethod
    def _cmp_labels(label1, label2):
        # unknown > not_unknown
        # otherwise, just use regular cmp
        u1 = isinstance(label1, Unknown)
        u2 = isinstance(label2, Unknown)
        if u1 == u2:
            return cmp(label1, label2)
        return u1 - u2

    def labels(self, unknown=True):
        """List of labels

        Parameters
        ----------
        unknown : bool, optional
            When False, do not return Unknown instances
            When True, return any label (even Unknown instances)

        Returns
        -------
        labels : list
            Sorted list of labels
        """

        if any([lnu for lnu in self._labelNeedsUpdate.values()]):
            self._updateLabels()

        labels = sorted(self._labels, cmp=self._cmp_labels)

        if not unknown:
            labels = [l for l in labels if not isinstance(l, Unknown)]

        return labels

    def get_labels(self, segment, unknown=True, unique=True):
        """Local set of labels

        Parameters
        ----------
        segment : Segment
            Segments to get label from.
        unknown : bool, optional
            When False, do not return Unknown instances
            When True, return any label (even Unknown instances)
        unique : bool, optional
            When False, return the list of (possibly repeated) labels.
            When True (default), return the set of labels
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

        if not unknown:
            labels = [l for l in labels if not isinstance(l, Unknown)]

        if unique:
            labels = set(labels)

        return labels

    def subset(self, labels, invert=False):
        """Annotation subset

        Extract annotation subset based on labels

        Parameters
        ----------
        labels : iterable
            Label iterable.
        invert : bool, optional
            If invert is True, extract all but requested `labels`

        Returns
        -------
        subset : `Annotation`
            Annotation subset.
        """

        labels = set(labels)

        if invert:
            labels = set(self.labels()) - labels
        else:
            labels = labels & set(self.labels())

        sub = self.__class__(uri=self.uri, modality=self.modality)
        for segment, track, label in self.itertracks(label=True):
            if label in labels:
                sub[segment, track] = label

        return sub

    def update(self, annotation):
        """Update existing annotations or create new ones

        Parameters
        ----------
        annotation : Annotation
            Updated (or new) annotations

        Returns
        -------
        self : Annotation
            Updated annotations.
        """

        for segment, track, label in annotation.itertracks(label=True):
            self[segment, track] = label
        return self

    def label_timeline(self, label):
        """Get timeline for a given label

        Parameters
        ----------
        label :

        Returns
        -------
        timeline : :class:`Timeline`
            Timeline made of all segments annotated with `label`

        """
        if label not in self.labels():
            return Timeline(uri=self.uri)

        if self._labelNeedsUpdate[label]:
            self._updateLabels()

            for l, hasChanged in self._labelNeedsUpdate.iteritems():
                if hasChanged:
                    self._labels[l] = Timeline(uri=self.uri)

            for segment, track, l in self.itertracks(label=True):
                if self._labelNeedsUpdate[l]:
                    self._labels[l].add(segment)

            self._labelNeedsUpdate = {l: False for l in self._labels}

        return self._labels[label].copy()

    def label_coverage(self, label):
        """

        Parameters
        ----------
        label :

        Returns
        -------

        """
        if label not in self.labels():
            return Timeline(uri=self.uri)

        return self.label_timeline(label).coverage()

    def label_duration(self, label):

        if label not in self.labels():
            return 0.

        return self.label_timeline(label).duration()

    def chart(self, percent=False):
        """
        Label chart based on their duration

        Parameters
        ----------
        percent : bool, optional
            Return total duration percentage (rather than raw duration)

        Returns
        -------
        chart : (label, duration) iterable
            Sorted from longest to shortest.

        """

        chart = sorted([(label, self.label_duration(label))
                        for label in self.labels()],
                       key=lambda x: x[1], reverse=True)

        if percent:
            total = np.sum([duration for _, duration in chart])
            chart = [(label, duration / total) for (label, duration) in chart]

        return chart

    def argmax(self, segment=None, known_first=False):
        """Get most frequent label


        Parameters
        ----------
        segment : Segment, optional
            Section of annotation where to look for the most frequent label.
            Defaults to whole annotation extent.
        known_first: bool, optional
            If True, artificially reduces the duration of intersection of
            `Unknown` labels so that 'known' labels are returned first.

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
            >>> if not annotation.argmax(segment):
            ...    print "No label intersecting %s" % segment
            No label intersection [22 --> 23]

        """

        # if annotation is empty, obviously there is no most frequent label
        if not self:
            return None

        # if segment is not provided, just look for the overall most frequent
        # label (ie. set segment to the extent of the annotation)
        if segment is None:
            segment = self.get_timeline().extent()

        # compute intersection duration for each label
        durations = {lbl: self.label_timeline(lbl)
                     .crop(segment, mode='intersection').duration()
                     for lbl in self.labels()}

        # artifically reduce intersection duration of Unknown labels
        # so that 'known' labels are returned first
        if known_first:
            maxduration = max(durations.values())
            for lbl in durations.keys():
                if isinstance(lbl, Unknown):
                    durations[lbl] = durations[lbl] - maxduration

        # find the most frequent label
        label = max(durations.iteritems(), key=operator.itemgetter(1))[0]

        # in case all durations were zero, there is no most frequent label
        return label if durations[label] > 0 else None

    def translate(self, translation):
        """Translate labels

        Parameters
        ----------
        translation: dict
            Label translation.
            Labels with no associated translation are kept unchanged.

        Returns
        -------
        translated : :class:`Annotation`
            New annotation with translated labels.
        """

        assert isinstance(translation, dict)

        # create an empty copy
        translated = self.empty()

        for segment, track, label in self.itertracks(label=True):
            # only transform labels that have an actual translation
            # in the provided dictionary, keep the others as they are.
            translated[segment, track] = translation.get(label, label)

        return translated

    def __mod__(self, translation):
        return self.translate(translation)

    def anonymize_labels(self):
        """Anonmyize labels

        Create a new annotation where labels are anonymized, ie. each label
        is replaced by a unique `Unknown` instance.

        Returns
        -------
        anonymized : :class:`Annotation`
            New annotation with anonymized labels.

        """
        translation = {label: Unknown() for label in self.labels()}
        return self % translation

    def anonymize_tracks(self):
        """
        Anonymize tracks

        Create a new annotation where each track is anonymized, i.e. the label
        of each track is set to a unique `Unknown` instance

        Returns
        -------
        anonymized : `Annotation`
            Anonymized annotation

        """
        anonymized = self.empty()
        for s, t, _ in self.itertracks(label=True):
            anonymized[s, t] = Unknown()
        return anonymized

    def smooth(self, collar=0.):
        """Smooth annotation

        Create new annotation where contiguous tracks with same label are
        merged into one longer track.

        Parameters
        ----------
        collar : float
            If collar is positive, also merge tracks separated by less than
            collar duration.

        Returns
        -------
        annotation : Annotation
            New annotation where contiguous tracks with same label are merged
            into one long track.

        Remarks
        -------
            Track names are lost in the process.
        """

        # initialize an empty annotation
        # with same uri and modality as original
        smoothed = self.empty()
        for label in self.labels():

            # get timeline for current label
            timeline = self.label_timeline(label)

            # fill the gaps shorter than collar
            if collar > 0.:
                gaps = timeline.gaps()
                for gap in gaps:
                    if gap.duration < collar:
                        timeline.add(gap)

            # reconstruct annotation with merged tracks
            for segment in timeline.coverage():
                track = smoothed.new_track(segment)
                smoothed[segment, track] = label

        # return
        return smoothed

    def co_iter(self, other):
        """
        Parameters
        ----------
        other : Annotation

        Generates
        ---------
        (segment, track), (other_segment, other_track)
        """

        for s, S in self.get_timeline().co_iter(other.get_timeline()):
            tracks = self.get_tracks(s)
            other_tracks = other.get_tracks(S)
            for t, T in itertools.product(tracks, other_tracks):
                yield (s, t), (S, T)

    def for_json(self):

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
        from pyannote.core.notebook import repr_annotation
        return repr_annotation(self)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
