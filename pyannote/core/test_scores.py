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


from scores import Scores
from segment import Segment
import numpy as np


class Test_Scores:

    def setup(self):

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

        self.scores = scores

    def test_get(self):
        segment, track = Segment(0., 2.5), 'track'
        assert self.scores[segment, track, 'A'] == 0.2
        assert self.scores[segment, track, 'B'] == 0.3
        assert self.scores[segment, track, 'C'] == 0.4
        assert self.scores[segment, track, 'D'] == 0.1

        segment, track = Segment(3., 4.), 'track'
        assert self.scores[segment, track, 'A'] == 0.4
        assert self.scores[segment, track, 'B'] == 0.5

        segment, track = Segment(3., 4.), 'other_track'
        assert self.scores[segment, track, 'B'] == 0.1
        assert self.scores[segment, track, 'C'] == 0.3
        assert self.scores[segment, track, 'D'] == 0.1

    def test_set(self):
        segment, track = Segment(0., 2.5), 'track'
        self.scores[segment, track, 'A'] = 1.
        assert self.scores[segment, track, 'A'] == 1.

    def test_set_newsegment(self):
        segment, track = Segment(0., 10.), 'track'
        self.scores[segment, track, 'A'] = 1.
        assert self.scores[segment, track, 'A'] == 1.
        assert np.isnan(self.scores[segment, track, 'B'])
        assert np.isnan(self.scores[segment, track, 'C'])
        assert np.isnan(self.scores[segment, track, 'D'])

    def test_set_newtrack(self):
        segment, track = Segment(0., 2.5), 'new_track'
        self.scores[segment, track, 'A'] = 1.
        assert self.scores[segment, track, 'A'] == 1.
        assert np.isnan(self.scores[segment, track, 'B'])
        assert np.isnan(self.scores[segment, track, 'C'])
        assert np.isnan(self.scores[segment, track, 'D'])

    def test_set_newlabel(self):
        segment, track = Segment(0., 2.5), 'track'
        self.scores[segment, track, 'E'] = 1.
        segment, track = Segment(3., 4.), 'track'
        assert np.isnan(self.scores[segment, track, 'E'])
        segment, track = Segment(3., 4.), 'other_track'
        assert np.isnan(self.scores[segment, track, 'E'])

    def test_del(self):
        segment, track = Segment(0., 2.5), 'track'
        del self.scores[segment, track]
        assert not self.scores.has_track(segment, track)
        assert not segment in self.scores

        segment, track = Segment(3., 4.), 'other_track'
        del self.scores[segment, track]
        assert not self.scores.has_track(segment, track)
        assert segment in self.scores
        segment, track = Segment(3., 4.), 'track'
        assert self.scores.has_track(segment, track)
