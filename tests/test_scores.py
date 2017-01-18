#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2014-2015 CNRS

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

import pytest

from pyannote.core import Scores
from pyannote.core import Segment
from pyannote.core.scores import Unknown
import numpy as np


@pytest.fixture
def scores():

    scores = Scores(uri='uri', modality='modality')

    segment, track = Segment(0., 2.5), 'track'
    scores[segment, track, 'A'] = 0.2
    scores[segment, track, 'B'] = 0.3
    scores[segment, track, 'C'] = 0.4
    scores[segment, track, 'D'] = 0.1

    segment, track = Segment(3., 4.), 'track'
    scores[segment, track, 'A'] = 0.4
    scores[segment, track, 'B'] = 0.5

    segment, track = Segment(3., 4.), 'other_track'
    scores[segment, track, 'B'] = 0.1
    scores[segment, track, 'C'] = 0.3
    scores[segment, track, 'D'] = 0.1

    return scores

def test_itertracks(scores):

    assert list(scores.itertracks()) == [(Segment(0, 2.5), 'track'),
                                         (Segment(3, 4), 'other_track'),
                                         (Segment(3, 4), 'track')]

def test_itervalues(scores):

    assert list(scores.itervalues()) == [(Segment(0, 2.5), 'track', 'A', 0.2),
                                         (Segment(0, 2.5), 'track', 'B', 0.3),
                                         (Segment(0, 2.5), 'track', 'C', 0.4),
                                         (Segment(0, 2.5), 'track', 'D', 0.1),
                                         (Segment(3, 4), 'other_track', 'B', 0.1),
                                         (Segment(3, 4), 'other_track', 'C', 0.3),
                                         (Segment(3, 4), 'other_track', 'D', 0.1),
                                         (Segment(3, 4), 'track', 'A', 0.4),
                                         (Segment(3, 4), 'track', 'B', 0.5)]

def test_get_track_scores(scores):

    assert scores.get_track_scores(Segment(0, 2.5), 'track') == {'A': 0.2,
                                                                 'B': 0.3,
                                                                 'C': 0.4,
                                                                 'D': 0.1}

    assert np.isnan(scores.get_track_scores(Segment(3, 4), 'other_track')['A'])

def test_labels(scores):

    assert scores.labels() == ['A', 'B', 'C', 'D']


def test_retrack(scores):

    assert list(scores.retrack().itertracks()) == [(Segment(0, 2.5), 0),
                                                   (Segment(3, 4), 1),
                                                   (Segment(3, 4), 2)]

# def test_apply(scores):
#     sum0 = scores.apply(np.sum, axis=0)
#     assert list(sum0.itervalues()) == False
#
#     sum1 = scores.apply(np.sum, axis=1)
#     assert list(sum1.itertracks()) == False

def test_rank(scores):

    ranked = scores.rank(ascending=False)
    assert list(ranked.itervalues()) ==  [(Segment(0, 2.5), 'track', 'A', 2),
                                          (Segment(0, 2.5), 'track', 'B', 1),
                                          (Segment(0, 2.5), 'track', 'C', 0),
                                          (Segment(0, 2.5), 'track', 'D', 3),
                                          (Segment(3, 4), 'other_track', 'B', 1.5),
                                          (Segment(3, 4), 'other_track', 'C', 0),
                                          (Segment(3, 4), 'other_track', 'D', 1.5),
                                          (Segment(3, 4), 'track', 'A', 1),
                                          (Segment(3, 4), 'track', 'B', 0)]

    ranked = scores.rank(ascending=True)
    assert list(ranked.itervalues()) ==  [(Segment(0, 2.5), 'track', 'A', 1),
                                          (Segment(0, 2.5), 'track', 'B', 2),
                                          (Segment(0, 2.5), 'track', 'C', 3),
                                          (Segment(0, 2.5), 'track', 'D', 0),
                                          (Segment(3, 4), 'other_track', 'B', 0.5),
                                          (Segment(3, 4), 'other_track', 'C', 2),
                                          (Segment(3, 4), 'other_track', 'D', 0.5),
                                          (Segment(3, 4), 'track', 'A', 0),
                                          (Segment(3, 4), 'track', 'B', 1)]


def test_nbest(scores):

    best1 = scores.nbest(1)
    assert list(best1.itervalues()) ==  [(Segment(0, 2.5), 'track', 'C', 0.4),
                                          (Segment(3, 4), 'other_track', 'C', 0.3),
                                          (Segment(3, 4), 'track', 'B', 0.5)]

def test_subset(scores):

    subset = scores.subset({'A', 'C'})
    assert list(subset.itervalues()) == [(Segment(0, 2.5), 'track', 'A', 0.2),
                                         (Segment(0, 2.5), 'track', 'C', 0.4),
                                         (Segment(3, 4), 'other_track', 'C', 0.3),
                                         (Segment(3, 4), 'track', 'A', 0.4)]

    isubset = scores.subset({'A', 'C'}, invert=True)
    assert list(isubset.itervalues()) == [(Segment(0, 2.5), 'track', 'B', 0.3),
                                          (Segment(0, 2.5), 'track', 'D', 0.1),
                                          (Segment(3, 4), 'other_track', 'B', 0.1),
                                          (Segment(3, 4), 'other_track', 'D', 0.1),
                                          (Segment(3, 4), 'track', 'B', 0.5)]

def test_to_annotation(scores):

    annotation = scores.to_annotation()
    assert list(annotation.itertracks(label=True)) == [(Segment(0, 2.5), 'track', 'C'),
                                                       (Segment(3, 4), 'other_track', 'C'),
                                                       (Segment(3, 4), 'track', 'B')]

    annotation = scores.to_annotation(threshold=0.4)
    assert isinstance(annotation[Segment(0, 2.5), 'track'], Unknown)
    assert isinstance(annotation[Segment(3, 4), 'other_track'], Unknown)
    assert annotation[Segment(3, 4), 'track'] == 'B'

    annotation = scores.to_annotation(posterior=True)
    assert annotation[Segment(0, 2.5), 'track'] == 'C'
    assert isinstance(annotation[Segment(3, 4), 'other_track'], Unknown)
    assert annotation[Segment(3, 4), 'track'] == 'B'

def test_map(scores):

    mapped = scores.map(lambda x: 2 * x)
    assert list(mapped.itervalues()) == [(Segment(0, 2.5), 'track', 'A', 0.4),
                                         (Segment(0, 2.5), 'track', 'B', 0.6),
                                         (Segment(0, 2.5), 'track', 'C', 0.8),
                                         (Segment(0, 2.5), 'track', 'D', 0.2),
                                         (Segment(3, 4), 'other_track', 'B', 0.2),
                                         (Segment(3, 4), 'other_track', 'C', 0.6),
                                         (Segment(3, 4), 'other_track', 'D', 0.2),
                                         (Segment(3, 4), 'track', 'A', 0.8),
                                         (Segment(3, 4), 'track', 'B', 1.0)]

def test_crop(scores):

    cropped = scores.crop(Segment(2, 4), mode='strict')
    assert list(cropped.itervalues()) == [(Segment(3, 4), 'other_track', 'B', 0.1),
                                         (Segment(3, 4), 'other_track', 'C', 0.3),
                                         (Segment(3, 4), 'other_track', 'D', 0.1),
                                         (Segment(3, 4), 'track', 'A', 0.4),
                                         (Segment(3, 4), 'track', 'B', 0.5)]

    cropped = scores.crop(Segment(2, 4), mode='loose')
    assert list(cropped.itervalues()) == [(Segment(0, 2.5), 'track', 'A', 0.2),
                                         (Segment(0, 2.5), 'track', 'B', 0.3),
                                         (Segment(0, 2.5), 'track', 'C', 0.4),
                                         (Segment(0, 2.5), 'track', 'D', 0.1),
                                         (Segment(3, 4), 'other_track', 'B', 0.1),
                                         (Segment(3, 4), 'other_track', 'C', 0.3),
                                         (Segment(3, 4), 'other_track', 'D', 0.1),
                                         (Segment(3, 4), 'track', 'A', 0.4),
                                         (Segment(3, 4), 'track', 'B', 0.5)]

def test_str(scores):

    assert str(scores) == (
        "                                         A    B    C    D\n"
        "segment_start segment_end track                          \n"
        "0             2.5         track        0.2  0.3  0.4  0.1\n"
        "3             4.0         other_track  NaN  0.1  0.3  0.1\n"
        "                          track        0.4  0.5  NaN  NaN")

def test_get(scores):
    segment, track = Segment(0., 2.5), 'track'
    assert scores[segment, track, 'A'] == 0.2
    assert scores[segment, track, 'B'] == 0.3
    assert scores[segment, track, 'C'] == 0.4
    assert scores[segment, track, 'D'] == 0.1

    segment, track = Segment(3., 4.), 'track'
    assert scores[segment, track, 'A'] == 0.4
    assert scores[segment, track, 'B'] == 0.5

    segment, track = Segment(3., 4.), 'other_track'
    assert scores[segment, track, 'B'] == 0.1
    assert scores[segment, track, 'C'] == 0.3
    assert scores[segment, track, 'D'] == 0.1


def test_set(scores):
    segment, track = Segment(0., 2.5), 'track'
    scores[segment, track, 'A'] = 1.
    assert scores[segment, track, 'A'] == 1.


def test_set_newsegment(scores):
    segment, track = Segment(0., 10.), 'track'
    scores[segment, track, 'A'] = 1.
    assert scores[segment, track, 'A'] == 1.
    assert np.isnan(scores[segment, track, 'B'])
    assert np.isnan(scores[segment, track, 'C'])
    assert np.isnan(scores[segment, track, 'D'])


def test_set_newtrack(scores):
    segment, track = Segment(0., 2.5), 'new_track'
    scores[segment, track, 'A'] = 1.
    assert scores[segment, track, 'A'] == 1.
    assert np.isnan(scores[segment, track, 'B'])
    assert np.isnan(scores[segment, track, 'C'])
    assert np.isnan(scores[segment, track, 'D'])


def test_set_newlabel(scores):
    segment, track = Segment(0., 2.5), 'track'
    scores[segment, track, 'E'] = 1.
    segment, track = Segment(3., 4.), 'track'
    assert np.isnan(scores[segment, track, 'E'])
    segment, track = Segment(3., 4.), 'other_track'
    assert np.isnan(scores[segment, track, 'E'])


def test_del(scores):
    segment, track = Segment(0., 2.5), 'track'
    del scores[segment, track]
    assert not scores.has_track(segment, track)
    assert not segment in scores

    segment, track = Segment(3., 4.), 'other_track'
    del scores[segment, track]
    assert not scores.has_track(segment, track)
    assert segment in scores
    segment, track = Segment(3., 4.), 'track'
    assert scores.has_track(segment, track)
