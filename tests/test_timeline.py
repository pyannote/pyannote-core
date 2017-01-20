import pytest

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


def test_iteration(timeline):

    assert list(timeline) == [Segment(0.5, 3),
                              Segment(1, 4),
                              Segment(5, 7),
                              Segment(6, 8),
                              Segment(7, 8),
                              Segment(8.5, 10)]

def test_getter(timeline):

    assert len(timeline) == 6
    assert str(timeline[1]) == "[ 00:00:01.000 -->  00:00:04.000]"


def test_extent(timeline):

    assert timeline.extent() == Segment(0.5, 10)

def test_coverage(timeline):

    assert list(timeline.coverage()) == [Segment(0.5, 4),
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
