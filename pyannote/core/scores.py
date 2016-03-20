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

import numpy as np
from pandas import Index, MultiIndex, DataFrame, pivot_table

from . import PYANNOTE_SEGMENT, PYANNOTE_TRACK, PYANNOTE_LABEL, PYANNOTE_SCORE
from .annotation import Annotation, Unknown
from .segment import Segment
from .timeline import Timeline


class Scores(object):
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
        dataframe = pivot_table(
            df, values=PYANNOTE_SCORE,
            index=[PYANNOTE_SEGMENT, PYANNOTE_TRACK], columns=PYANNOTE_LABEL,
            aggfunc=aggfunc
        )

        annotation = Annotation(uri=uri, modality=modality)
        for index, _ in dataframe.iterrows():
            segment = Segment(*index[0])
            track = index[1]
            annotation[segment, track] = ''

        labels = dataframe.columns

        return cls(uri=uri, modality=modality,
                   annotation=annotation, labels=labels,
                   values=dataframe.values)

    def __init__(self, uri=None, modality=None,
                 annotation=None, labels=None,
                 values=None, dtype=None):

        super(Scores, self).__init__()

        names = [PYANNOTE_SEGMENT + '_' + field
                 for field in Segment._fields] + [PYANNOTE_TRACK]

        if annotation:
            annotation = annotation.copy()
            index = Index(
                [s + (t, ) for s, t in annotation.itertracks()],
                name=names)

        else:
            annotation = Annotation(uri=uri, modality=modality)
            index = MultiIndex(levels=[list() for name in names],
                               labels=[list() for name in names],
                               names=names)

        self.annotation_ = annotation
        columns = None if labels is None else list(labels)
        data = None if values is None else np.array(values)
        dtype = np.float if values is None else values.dtype

        self.dataframe_ = DataFrame(data=data, dtype=dtype,
                                    index=index, columns=columns)

        self.hasChanged_ = True

        self.modality = modality
        self.uri = uri

    def copy(self):
        self._reindexIfNeeded()
        copied = self.__class__(uri=self.uri, modality=self.modality)
        copied.dataframe_ = self.dataframe_.copy()
        copied.annotation_ = self.annotation_.copy()
        copied.hasChanged_ = self.hasChanged_
        return copied

    # del scores[segment]
    # del scores[segment, :]
    # del scores[segment, track]
    def __delitem__(self, key):

        if isinstance(key, Segment):
            segment = key
            self.dataframe_.drop(tuple(segment), axis=0, inplace=True)
            del self.annotation_[segment]
            self.hasChanged_ = True

        elif isinstance(key, tuple) and len(key) == 2:
            segment, track = key
            self.dataframe_.drop(tuple(segment) + (track, ),
                                 axis=0, inplace=True)
            del self.annotation_[segment, track]
            self.hasChanged_ = True

        else:
            raise KeyError('')

    # value = scores[segment, track, label]
    def __getitem__(self, key):

        if len(key) == 2:
            key = (key[0], '_', key[1])

        segment, track, label = key
        return self.dataframe_.at[tuple(segment) + (track, ), label]

    # scores[segment, track, label] = value
    # scores[segment, label] ==== scores[segment, '_', label]
    def __setitem__(self, key, value):

        if len(key) == 2:
            key = (key[0], '_', key[1])

        segment, track, label = key

        # do not add empty track
        if not segment:
            return

        self.dataframe_.at[tuple(segment) + (track,), label] = value
        self.annotation_[segment, track] = label
        self.hasChanged_ = True

    def __len__(self):
        """Number of annotated segments"""
        return len(self.annotation_)

    def __nonzero__(self):
        return self.__bool__()

    def __bool__(self):
        """False if annotation is empty"""
        return True if self.annotation_ else False

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
        return included in self.annotation_

    def __iter__(self):
        """Iterate over sorted segments"""
        return iter(self.annotation_.get_timeline())

    def __reversed__(self):
        """Reverse iterate over sorted segments"""
        return reversed(self.annotation_.get_timeline())

    def itersegments(self):
        return iter(self)

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
        return self.annotation_.get_tracks(segment)

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
        return self.annotation_.has_track(segment, track)

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
        return self.annotation_.get_track_by_name(track)

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

        return self.annotation_.new_track(segment, candidate=None, prefix=None)

    def itertracks(self):
        """Iterate over annotation as (segment, track) tuple"""
        return self.annotation_.itertracks()

    def itervalues(self):
        """Iterate over scores as (segment, track, label, value) tuple"""

        # make sure segment/track pairs are sorted
        self._reindexIfNeeded()

        labels = self.labels()

        # yield one (segment, track, label) tuple per loop
        for index, columns in self.dataframe_.iterrows():
            segment = Segment(*index[:-1])
            track = index[-1]
            for label in labels:
                value = columns[label]
                if not np.isnan(value):
                    yield segment, track, label, value

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
        return dict(self.dataframe_.xs(tuple(segment) + (track, )))

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
        labels = sorted(self.dataframe_.columns, key=str)
        if unknown:
            return labels
        else:
            return [l for l in labels if not isinstance(l, Unknown)]

    def _reindexIfNeeded(self):

        if not self.hasChanged_:
            return

        names = [PYANNOTE_SEGMENT + '_' + field
                 for field in Segment._fields] + [PYANNOTE_TRACK]

        new_index = Index(
            [s + (t, ) for s, t in self.annotation_.itertracks()],
            name=names)

        self.dataframe_ = self.dataframe_.reindex(new_index)

        self.hasChanged_ = False

        return

    def retrack(self):
        """
        """

        self._reindexIfNeeded()
        retracked = self.copy()

        annotation = self.annotation_.retrack()
        retracked.annotation_ = annotation

        names = [PYANNOTE_SEGMENT + '_' + field
                 for field in Segment._fields] + [PYANNOTE_TRACK]
        new_index = Index(
            [s + (t, ) for s, t in annotation.itertracks()],
            name=names)
        retracked.dataframe_.index = new_index

        return retracked

    def apply(self, func, axis=0):

        applied = self.copy()
        applied.dataframe_ = self.dataframe_.apply(func, axis=axis)
        applied.hasChanged_ = True

        return applied

    def rank(self, ascending=False):
        """

        Parameters
        ----------
        ascending : boolean, default False
            False for ranks by high (0) to low (N-1)

        Returns
        -------
        rank : `Scores`

        """

        ranked = self.copy()
        ranked.dataframe_ = -1 + self.dataframe_.rank(axis=1,
                                                      ascending=ascending)
        ranked.hasChanged_ = True
        return ranked

    def nbest(self, n, ascending=False):
        """

        Parameters
        ----------
        n : int
            Size of n-best list
        ascending : boolean, default False
            False for ranks by high (0) to low (N-1)

        Returns
        -------
        nbest : `Scores`
            New scores where only n-best are kept.

        """

        filtered = self.copy()
        ranked_ = -1 + self.dataframe_.rank(axis=1, ascending=ascending)
        filtered.dataframe_ = filtered.dataframe_.where(ranked_ < n,
                                                        other=np.NaN)
        filtered.hasChanged_ = True
        return filtered

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

        self._reindexIfNeeded()

        if not isinstance(labels, set):
            raise TypeError('labels must be provided as a set of labels.')

        if invert:
            labels = set(self.labels()) - labels
        else:
            labels = labels & set(self.labels())

        subset = Scores(uri=self.uri, modality=self.modality)
        subset.annotation_ = self.annotation_
        subset.dataframe_ = self.dataframe_[list(labels)]

        return subset

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

        if not self:
            return Annotation(uri=self.uri, modality=self.modality)

        best = self.nbest(1, ascending=False)
        large_enough = best.copy()

        if posterior:
            unknown_posterior = 1. - self.dataframe_.sum(axis=1)

            large_enough.dataframe_ = (
                ((best.dataframe_.T > unknown_posterior) &
                 (best.dataframe_.T > threshold)).T
            )

        else:

            large_enough.dataframe_ = (
                (best.dataframe_.T > threshold).T
            )

        large_enough.dataframe_.where(best.dataframe_.notnull(),
                                      inplace=True, other=np.NaN)

        annotation = Annotation(uri=self.uri, modality=self.modality)
        for segment, track, label, value in large_enough.itervalues():
            label = label if value else Unknown()
            annotation[segment, track] = label

        return annotation

    def map(self, func):
        """Apply function to all values"""

        mapped = self.copy()
        mapped.dataframe_ = self.dataframe_.applymap(func)
        mapped.hasChanged_ = True
        return mapped

    def crop(self, focus, mode='strict'):
        """Crop on focus

        Parameters
        ----------
        focus : `Segment` or `Timeline`

        mode : {'strict', 'loose', 'intersection'}
            In 'strict' mode, only segments fully included in focus coverage
            are kept. In 'loose' mode, any intersecting segment is kept
            unchanged. In 'intersection' mode, only intersecting segments are
            kept and replaced by their actual intersection with the focus.

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

        self._reindexIfNeeded()
        cropped = self.copy()

        if mode in ['strict', 'loose']:

            new_annotation = self.annotation_.crop(focus, mode=mode)
            keep = [new_annotation.has_track(segment, track)
                    for segment, track in self.itertracks()]
            cropped.dataframe_ = self.dataframe_[keep]
            cropped.annotation_ = new_annotation
            cropped.hasChanged_ = True

            return cropped

        elif mode in ['intersection']:

            raise NotImplementedError('')

            # # two original segments might be cropped into the same resulting
            # # segment -- therefore, we keep track of the mapping
            # intersection, mapping = timeline.crop(coverage,
            #                                       mode=mode, mapping=True)
            #
            # # create new empty annotation
            # A = self.__class__(uri=self.uri, modality=self.modality)
            #
            # for cropped in intersection:
            #     for original in mapping[cropped]:
            #         for track in self.tracks(original):
            #             # try to use original track name (candidate)
            #             # if it already exists, create a brand new one
            #             new_track = A.new_track(cropped, candidate=track)
            #             # copy each value, column by column
            #             for label in self.dataframe_.columns:
            #                 value = self.dataframe_.get_value((original, track),
            #                                            label)
            #                 A.dataframe_ = A.dataframe_.set_value((cropped, new_track),
            #                                         label, value)
            #
            # return A

    def __str__(self):
        """Human-friendly representation"""
        if self:
            self._reindexIfNeeded()
            return str(self.dataframe_)
        else:
            return ""

    def _repr_png_(self):
        from .notebook import repr_scores
        return repr_scores(self)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
