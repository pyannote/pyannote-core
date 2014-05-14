#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2014 CNRS (Hervé BREDIN - http://herve.niderb.fr)

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import unicode_literals

import numpy as np
from pandas import MultiIndex, DataFrame, pivot_table

from . import PYANNOTE_SEGMENT, PYANNOTE_TRACK, PYANNOTE_LABEL, PYANNOTE_SCORE
from annotation import Annotation, Unknown
from segment import Segment
from timeline import Timeline


class AnnotationMixin(object):

    def get_timeline(self):
        if self._timelineHasChanged:
            self._timeline = Timeline(segments=[s for s, _ in self._df.index],
                                      uri=self.uri)
            self._timelineHasChanged = False
        return self._timeline

    def __len__(self):
        """Number of annotated segments"""
        return len(self.get_timeline())

    def __nonzero__(self):
        """False if annotation is empty"""
        return len(self.get_timeline()) > 0

    def __contains__(self, included):
        """Check if segments are annotated

        Parameters
        ----------
        included : `Segment` or `Timeline`

        Returns
        -------
        contains : bool
            True if every segment in `included` is annotated, False otherwise.
        """
        return included in self.get_timeline()

    def __iter__(self):
        """Iterate over sorted segments"""
        return iter(self.get_timeline())

    def __reversed__(self):
        """Reverse iterate over sorted segments"""
        return reversed(self.get_timeline())

    def itersegments(self):
        return iter(self.get_timeline())

    def itertracks(self):
        """Iterate over annotation as (segment, track) tuple"""

        # make sure segment/track pairs are sorted
        self._df = self._df.sort_index()

        for (segment, track), _ in self._df.iterrows():
            yield segment, track

    def crop(self, focus, mode='strict'):
        """Crop on focus

        Parameters
        ----------
        focus : `Segment` or `Timeline`

        mode : {'strict', 'loose', 'intersection'}
            In 'strict' mode, only segments fully included in focus coverage are
            kept. In 'loose' mode, any intersecting segment is kept unchanged.
            In 'intersection' mode, only intersecting segments are kept and
            replaced by their actual intersection with the focus.

        Returns
        -------
        cropped : same type as caller
            Cropped version of the caller containing only tracks matching
            the provided focus and mode.

        Remarks
        -------
        In 'intersection' mode, the best is done to keep the track names
        unchanged. However, in some cases where two original segments are
        cropped into the same resulting segments, conflicting track names are
        modified to make sure no track is lost.

        """
        if isinstance(focus, Segment):

            return self.crop(Timeline([focus], uri=self.uri), mode=mode)

        elif isinstance(focus, Timeline):

            # timeline made of all annotated segments
            timeline = self.get_timeline()

            # focus coverage
            # coverage = focus.coverage()
            coverage = focus.segmentation()

            if mode in ['strict', 'loose']:

                # segments (strictly or loosely) included in requested coverage
                included = timeline.crop(coverage, mode=mode)

                # boolean array: True if row must be kept, False otherwise
                keep = [(s in included) for s, _ in self._df.index]

                # crop-crop
                A = self.__class__(uri=self.uri, modality=self.modality)
                A._df = self._df[keep]

                return A

            elif mode == 'intersection':

                # two original segments might be cropped into the same resulting
                # segment -- therefore, we keep track of the mapping
                intersection, mapping = timeline.crop(coverage,
                                                      mode=mode, mapping=True)

                # create new empty annotation
                A = self.__class__(uri=self.uri, modality=self.modality)

                for cropped in intersection:
                    for original in mapping[cropped]:
                        for track in self.tracks(original):
                            # try to use original track name (candidate)
                            # if it already exists, create a brand new one
                            new_track = A.new_track(cropped, candidate=track)
                            # copy each value, column by column
                            for label in self._df.columns:
                                value = self._df.get_value((original, track),
                                                           label)
                                A._df = A._df.set_value((cropped, new_track),
                                                        label, value)

                return A

        else:
            raise TypeError('')

    def tracks(self, segment):
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

        try:
            df = self._df.xs(segment)
            existing_tracks = set(df.index)

        except Exception, e:
            existing_tracks = set([])

        return existing_tracks

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
        return (segment, track) in self._df.index

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
        try:
            segments = list(self._df.xs(track, level=1).index)
        except Exception, e:
            segments = []
        return [(s, track) for s in segments]

    def copy(self):
        A = self.__class__(uri=self.uri, modality=self.modality)
        A._df = self._df.copy()
        return A

    def retrack(self):
        """
        """
        A = self.copy()
        reindex = MultiIndex.from_tuples([(s, n)
                                          for n, (s, _) in enumerate(A._df.index)])
        A._df.index = reindex
        return A

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
        existing_tracks = self.tracks(segment)

        # if candidate is provided, check whether it already exists
        # in case it does not, use it
        if candidate is not None:
            if candidate not in existing_tracks:
                return candidate

        # no candidate was provided or the provided candidate already exists
        # we need to create a brand new one

        # by default (if prefix is not provided)
        # use modality as prefix (eg. speaker1, speaker2, ...)
        if prefix is None:
            prefix = '' if self.modality is None else str(self.modality)

        # find first non-existing track name for segment
        # eg. if speaker1 exists, try speaker2, then speaker3, ...
        count = 1
        while ('%s%d' % (prefix, count)) in existing_tracks:
            count += 1

        # return first non-existing track name
        return '%s%d' % (prefix, count)

    def __str__(self):
        """Human-friendly representation"""
        if self:
            self._df.sort_index(inplace=True)
            return str(self._df)
        else:
            return ""


class Scores(AnnotationMixin, object):
    """

    Parameters
    ----------
    uri : str, optional

    modality : str, optional

    Returns
    -------
    scores : `Scores`

    Examples
    --------

        >>> s = Scores(uri='video', modality='speaker')
        >>> s[Segment(0,1), 's1', 'A'] = 0.1
        >>> s[Segment(0,1), 's1', 'B'] = 0.2
        >>> s[Segment(0,1), 's1', 'C'] = 0.3
        >>> s[Segment(0,1), 's2', 'A'] = 0.4
        >>> s[Segment(0,1), 's2', 'B'] = 0.3
        >>> s[Segment(0,1), 's2', 'C'] = 0.2
        >>> s[Segment(2,3), 's1', 'A'] = 0.2
        >>> s[Segment(2,3), 's1', 'B'] = 0.1
        >>> s[Segment(2,3), 's1', 'C'] = 0.3

    """
    @classmethod
    def from_df(
        cls, df,
        uri=None, modality=None, aggfunc=np.mean
    ):
        """

        Parameters
        ----------
        df : DataFrame
            Must contain the following columns:
            'segment', 'track', 'label' and 'value'
        uri : str, optional
            Resource identifier
        modality : str, optional
            Modality
        aggfunc : func
            Value aggregation function in case of duplicate (segment, track,
            label) tuples

        Returns
        -------

        """
        A = cls(uri=uri, modality=modality)
        A._df = pivot_table(
            df, values=PYANNOTE_SCORE,
            rows=[PYANNOTE_SEGMENT, PYANNOTE_TRACK], cols=PYANNOTE_LABEL,
            aggfunc=aggfunc
        )
        return A

    def __init__(self, uri=None, modality=None):
        super(Scores, self).__init__()

        index = MultiIndex(
            levels=[[], []], labels=[[], []],
            names=[PYANNOTE_SEGMENT, PYANNOTE_TRACK]
        )

        self._df = DataFrame(index=index, dtype=np.float64)
        self.modality = modality
        self.uri = uri
        self._timelineHasChanged = True

    # del scores[segment]
    # del scores[segment, :]
    # del scores[segment, track]
    def __delitem__(self, key):

        if isinstance(key, Segment):
            segment = key
            self._df = self._df.drop(segment, axis=0)
            self._timelineHasChanged = True

        elif isinstance(key, tuple) and len(key) == 2:
            segment, track = key
            self._df = self._df.drop((segment, track), axis=0)
            self._timelineHasChanged = True

        else:
            raise KeyError('')

    # value = scores[segment, track, label]
    def __getitem__(self, key):
        segment, track, label = key
        return self._df.get_value((segment, track), label)

    def get_track_scores(self, segment, track):
        """Get all scores for a given track.

        Parameters
        ----------
        segment : Segment
        track : hashable
            segment, track must be a valid track

        Returns
        -------
        scores : dict
            {label: score} dictionary
        """
        return {l: self._df.get_value((segment, track), l) for l in self._df}

    # scores[segment, track, label] = value
    def __setitem__(self, key, value):
        segment, track, label = key
        self._df = self._df.set_value((segment, track), label, value)
        self._timelineHasChanged = True

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
            Sorted list of existing labels

        Remarks
        -------
            Labels are sorted based on their string representation.
        """
        labels = sorted(self._df.columns, key=str)
        if unknown:
            return labels
        else:
            return [l for l in labels if not isinstance(l, Unknown)]

    def itervalues(self):
        """Iterate over annotation as (segment, track, label, value) tuple"""

        # make sure segment/track pairs are sorted
        self._df = self._df.sort_index()

        # yield one (segment, track, label) tuple per loop
        labels = self._df.columns
        for (segment, track), columns in self._df.iterrows():
            for label in labels:
                value = columns[label]
                if np.isnan(value):
                    continue
                else:
                    yield segment, track, label, value

    def _rank(self, invert):

        if invert:
            direction = 1.

        else:
            direction = -1.

        def nan_rank(data):

            # replace NaN by -inf or +inf depending on the requested direction
            finite = np.isfinite(data)
            fixed = np.where(finite, direction*data, -direction*np.inf)

            # do the actual argsort
            indices = np.argsort(fixed)
            # get rank from argsort
            rank = np.argsort(indices)

            # special treatment for inverted NaN scores
            # (we want ranks to start at 0 even in case of NaN)
            if invert:
                rank = np.where(finite, rank-(len(data)-np.sum(finite)), np.nan)
            else:
                rank = np.where(finite, rank, np.nan)
            return rank

        return self._df.apply(nan_rank, axis=1)

    def rank(self, invert=False):
        """

        Parameters
        ----------
        invert : bool, optional
            By default, larger scores are better.
            Set `invert` to True to indicate smaller scores are better.

        Returns
        -------
        rank : `Scores`

        """
        A = self.__class__(uri=self.uri, modality=self.modality)
        A._df = self._rank(invert)
        return A

    def nbest(self, n, invert=False):
        """

        Parameters
        ----------
        n : int
            Size of n-best list
        invert : bool, optional
            By default, larger scores are better.
            Set `invert` to True to indicate smaller scores are better.

        Returns
        -------
        nbest : `Scores`
            New scores where only n-best are kept.

        """
        df = self._df.copy()
        nbest = self._rank(invert) < n
        df[~nbest] = np.nan

        A = self.__class__(uri=self.uri, modality=self.modality)
        A._df = df

        return A

    def subset(self, labels, invert=False):
        """Scores subset

        Extract scores subset based on labels

        Parameters
        ----------
        labels : set
            Set of labels
        invert : bool, optional
            If invert is True, extract all but requested `labels`

        Returns
        -------
        subset : `Scores`
            Scores subset.
        """

        if not isinstance(labels, set):
            raise TypeError('labels must be provided as a set of labels.')

        if invert:
            labels = set(self.labels()) - labels
        else:
            labels = labels & set(self.labels())

        A = self.__class__(uri=self.uri, modality=self.modality)
        A._df = self._df[list(labels)]

        return A

    def to_annotation(self, threshold=-np.inf, posterior=False):
        """

        Parameters
        ----------
        threshold : float, optional
            Each track is annotated with the label with the highest score.
            Yet, if the latter is smaller than `threshold`, label is replaced
            with an `Unknown` instance.
        posterior : bool, optional
            If True, scores are posterior probabilities in open-set
            identification. If top model posterior is higher than unknown
            posterior, it is selected. Otherwise, label is replaced with an
            `Unknown` instance.
        """

        annotation = Annotation(uri=self.uri, modality=self.modality)
        if not self:
            return annotation

        best = self.nbest(1, invert=False)

        if posterior:

            # compute unknown posterior
            func = lambda p: 1. - np.nansum(p, axis=1)
            Pu = self.apply(func, new_columns=['_'])

            # threshold best target posterior
            # with unknown posterior and threshold
            for segment, track, label, value in best.itervalues():

                if value < Pu[segment, track, '_'] or value < threshold:
                    label = Unknown()

                annotation[segment, track] = label

        else:

            # threshold best target score with threshold
            for segment, track, label, value in best.itervalues():
                if value < threshold:
                    label = Unknown()
                annotation[segment, track] = label

        return annotation

    def map(self, func):
        """Apply function to all values"""
        A = self.__class__(uri=self.uri, modality=self.modality)
        A._df = func(self._df)
        return A

    def apply(self, data_func, new_index=None, new_columns=None):
        """Apply `data_func` on internal numpy array

        Parameters
        ----------
        data_func : func
            Function expecting (index x columns) numpy array as input
        new_index : iterable, optional
            When provided, these will be the index of returned array.
        new_columns : iterable, optional
            When provided, these will be the columns of returned array.
        """
        new_data = data_func(self._df.values)

        if new_index is None:
            new_index = self._df.index

        if new_columns is None:
            new_columns = self._df.columns

        df = DataFrame(
            data=new_data,
            index=new_index,
            columns=new_columns)

        new_scores = self.__class__(uri=self.uri, modality=self.modality)
        new_scores._df = df

        return new_scores

    def _repr_png_(self):
        from pyannote.core.notebook import repr_scores
        return repr_scores(self)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
