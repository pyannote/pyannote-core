from pyannote.core import Segment

def test_creation():

    s = 1.
    e = 9.
    segment = Segment(start=s, end=e)

    assert str(segment) == "[1.000 --> 9.000]"
    assert segment.end == 9
    assert segment.duration == 8
    assert segment.middle == 5


def test_intersection():

    segment = Segment(start=1, end=9)
    other_segment = Segment(4, 13)

    assert segment.intersects(other_segment)
    assert segment & other_segment == Segment(4, 9)

    other_segment = Segment(13, 20)

    assert not segment.intersects(other_segment)


def test_inclusion():
    
    segment = Segment(start=1, end=9)
    other_segment = Segment(5, 9)

    print(other_segment in segment)
    assert other_segment in segment
    assert not segment.overlaps(23)


def test_other_operation():

    segment = Segment(start=1, end=9)
    other_segment = Segment(10, 30)
    assert segment | other_segment == Segment(1, 30)

    other_segment = Segment(14, 15)
    assert segment ^ other_segment == Segment(9, 14)
