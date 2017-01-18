from pyannote.core import Transcription
from pyannote.core import T
from pyannote.core import Segment

import pytest


@pytest.fixture
def transcription():
    transcription = Transcription()
    transcription.add_edge(3, 5, speech="hi there, I'm Penny",
                           speaker='Penny')
    transcription.add_edge(5, 5.5)
    transcription.add_edge(5.5, 7, speech="hi. I'm Leonard", speaker='Leonard')

    transcription.add_edge(T('A'), T('B'), summary='Leonard meets Penny')

    transcription.add_edge(7, T('C'))

    return transcription


def test_drifting(transcription):
    assert set(transcription.drifting()) == set(['A', 'B', 'C'])

def test_anchored(transcription):
    assert set(transcription.anchored()) == set([3, 5, 5.5, 7])

def test_relabel_drifting_nodes(transcription):
    # how can we test this?
    pass

def test_crop(transcription):
    cropped = transcription.crop(Segment(5.2, 8))
    assert list(cropped.ordered_edges_iter(data=False)) == [(5.0, 5.5),
                                                            (5.5, 7.0),
                                                            (7.0, 'C')]

def test_anchor(transcription):
    transcription.anchor('A', 2)
    assert list(transcription.ordered_edges_iter()) == [(2, 'B'),
                                                        (3, 5),
                                                        (5, 5.5),
                                                        (5.5, 7),
                                                        (7, 'C')]

def test_align(transcription):
    transcription.align('A', 'C')
    assert list(transcription.ordered_edges_iter()) == [(3, 5),
                                                        (5, 5.5),
                                                        (5.5, 7),
                                                        (7, 'C'),
                                                        ('C', 'B')]

def test_prealign(transcription):
    pass

def test_postalign(transcription):
    pass

def test_ordering_graph(transcription):
    assert sorted(transcription.ordering_graph().edges()) == [(3.0, 5.0),
                                                              (3.0, 5.5),
                                                              (3.0, 7.0),
                                                              (3.0, 'C'),
                                                              (5.0, 5.5),
                                                              (5.0, 7.0),
                                                              (5.0, 'C'),
                                                              (5.5, 7.0),
                                                              (5.5, 'C'),
                                                              (7.0, 'C'),
                                                              ('A', 'B')]

def test_temporal_sort(transcription):
    transcription.remove_node('A')
    transcription.remove_node('B')
    assert transcription.temporal_sort() == [3, 5, 5.5, 7, 'C']

def test_timerange(transcription):
    pass
