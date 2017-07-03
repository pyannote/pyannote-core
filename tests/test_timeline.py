#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2016-2017 CNRS

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

from pyannote.core import Annotation
from pyannote.core import Timeline
from pyannote.core import Segment

@pytest.fixture
def timeline():
    t = Timeline(uri='MyAudioFile')
    t.add(Segment(6, 8))
    t.add(Segment(0.5, 3))
    t.add(Segment(8.5, 10))
    t.add(Segment(1, 4))
    t.add(Segment(5, 7))
    t.add(Segment(7, 8))

    return t

def test_to_annotation(timeline):
    expected = Annotation(uri='MyAudioFile', modality='MyModality')
    expected[Segment(6, 8)] = 'D'
    expected[Segment(0.5, 3)] = 'A'
    expected[Segment(8.5, 10)] = 'F'
    expected[Segment(1, 4)] = 'B'
    expected[Segment(5, 7)] = 'C'
    expected[Segment(7, 8)] = 'E'
    assert expected == timeline.to_annotation(modality='MyModality')

def test_iteration(timeline):

    assert list(timeline) == [Segment(0.5, 3),
                              Segment(1, 4),
                              Segment(5, 7),
                              Segment(6, 8),
                              Segment(7, 8),
                              Segment(8.5, 10)]

def test_remove(timeline):
    timeline.remove(Segment(1, 4))
    timeline.remove(Segment(5, 7))
    assert list(timeline) == [Segment(0.5, 3),
                              Segment(6, 8),
                              Segment(7, 8),
                              Segment(8.5, 10)]

def test_getter(timeline):
    assert len(timeline) == 6
    assert str(timeline[1]) == "[ 00:00:01.000 -->  00:00:04.000]"

def test_getter_negative(timeline):
    assert timeline[-2] == Segment(7, 8)

def test_extent(timeline):
    assert timeline.extent() == Segment(0.5, 10)

def test_remove_and_extent():
    t = Timeline(uri='MyAudioFile')
    t.add(Segment(6, 8))
    t.add(Segment(7, 9))
    t.add(Segment(6, 9))

    t.remove(Segment(6, 9))
    assert t.extent() == Segment(6, 9)

def test_extent(timeline):
    assert timeline.extent() == Segment(0.5, 10)

def test_support(timeline):
    assert list(timeline.support()) == [Segment(0.5, 4),
                                        Segment(5, 8),
                                        Segment(8.5, 10)]

def test_gaps(timeline):
    assert list(timeline.gaps()) == [Segment(4, 5),
                                     Segment(8, 8.5)]

def test_crop(timeline):

    selection = Segment(3,7)

    expected_answer = Timeline(uri='MyAudioFile')
    expected_answer.add(Segment(3, 4))
    expected_answer.add(Segment(5, 7))
    expected_answer.add(Segment(6, 7))

    assert timeline.crop(selection, mode='intersection') == expected_answer

    expected_answer = Timeline(uri='MyAudioFile')
    expected_answer.add(Segment(5, 7))
    assert timeline.crop(selection, mode='strict') == expected_answer


    expected_answer = Timeline(uri="pouet")
    expected_answer.add(Segment(1, 4))
    expected_answer.add(Segment(5, 7))
    expected_answer.add(Segment(6, 8))

    timeline.crop(selection, mode='loose') == expected_answer

def test_crop_mapping():

    timeline = Timeline([Segment(0, 2), Segment(1, 2), Segment(3, 4)])
    cropped, mapping = timeline.crop(Segment(1, 2), returns_mapping=True)

    expected_cropped = Timeline([Segment(1, 2)])
    assert cropped == expected_cropped

    expected_mapping = {Segment(1, 2): [Segment(0, 2), Segment(1, 2)]}
    assert mapping == expected_mapping

def test_union():
    first_timeline = Timeline([Segment(0, 1),
                               Segment(2, 3),
                               Segment(4, 5)])
    second_timeline = Timeline([Segment(1.5, 4.5)])

    assert first_timeline.union(second_timeline) == Timeline([Segment(0, 1),
                                                              Segment(1.5, 4.5),
                                                              Segment(2, 3),
                                                              Segment(4, 5)])

    assert second_timeline.crop(first_timeline) == Timeline([Segment(2, 3),
                                                             Segment(4, 4.5)])

    assert list(first_timeline.co_iter(second_timeline)) == [(Segment(2, 3), Segment(1.5, 4.5)),
                                                             (Segment(4, 5), Segment(1.5, 4.5))]
def test_union_extent():
    first_timeline = Timeline([Segment(0, 1),
                               Segment(2, 3),
                               Segment(4, 5)])
    second_timeline = Timeline([Segment(1.5, 6)])

    union_timeline = first_timeline.union(second_timeline)
    assert union_timeline.extent() == Segment(0, 6)


def test_update_extent():
    timeline = Timeline([Segment(0, 1), Segment(2, 3), Segment(4, 5)])
    other_timeline = Timeline([Segment(1.5, 6)])
    timeline.update(other_timeline)
    assert timeline.extent() == Segment(0, 6)
