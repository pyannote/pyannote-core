import pytest

from pyannote.core import Annotation
from pyannote.core import Segment
from pyannote.core import Timeline


@pytest.fixture
def annotation():
    annotation = Annotation(
        uri='TheBigBangTheory.Season01.Episode01',
        modality='speaker')
    annotation[Segment(3, 5), '_'] = 'Penny'
    annotation[Segment(5.5, 7), '_',] = 'Leonard'
    annotation[Segment(8, 10), '_'] = 'Penny'
    annotation[Segment(8, 10), 'anything'] = 'Sheldon'

    return annotation

def test_copy(annotation):

    copy = annotation.copy()
    assert copy == annotation

def test_creation(annotation):

    assert list(annotation.itersegments()) == [Segment(3, 5),
                                               Segment(5.5, 7),
                                               Segment(8, 10)]

    assert list(annotation.itertracks()) == [(Segment(3, 5), '_'),
                                             (Segment(5.5, 7), '_'),
                                             (Segment(8, 10), '_'),
                                             (Segment(8, 10), 'anything')]

    assert list(annotation.itertracks(label=True)) == [(Segment(3, 5), '_', 'Penny'),
                                                       (Segment(5.5, 7), '_', 'Leonard'),
                                                       (Segment(8, 10), '_', 'Penny'),
                                                       (Segment(8, 10), 'anything', 'Sheldon')]

def test_segments(annotation):

    assert annotation.get_timeline(copy=False) == Timeline([Segment(3, 5),
                                                  Segment(5.5, 7),
                                                  Segment(8, 10)],
                                                 uri="TheBigBangTheory.Season01.Episode01")

def test_segments_copy(annotation):

    assert annotation.get_timeline(copy=True) == Timeline([Segment(3, 5),
                                                  Segment(5.5, 7),
                                                  Segment(8, 10)],
                                                 uri="TheBigBangTheory.Season01.Episode01")



def test_tracks(annotation):
    assert not annotation.has_track(Segment(8, 10), '---')
    assert annotation.get_tracks(Segment(8, 10)) == {'_', 'anything'}

    assert list(annotation.retrack().itertracks()) == [(Segment(3, 5), 0),
                                                       (Segment(5.5, 7), 1),
                                                       (Segment(8, 10), 2),
                                                       (Segment(8, 10), 3)]


def test_labels(annotation):

    assert annotation.labels() == ['Leonard', 'Penny', 'Sheldon']
    assert annotation.get_labels(Segment(8, 10)) == {'Penny', 'Sheldon'}

    expected_res = Annotation(
        uri='TheBigBangTheory.Season01.Episode01',
        modality='speaker')
    expected_res[Segment(3, 5), '_'] = 'Kaley Cuoco'
    expected_res[Segment(5.5, 7), '_',] = 'Johnny Galecki'
    expected_res[Segment(8, 10), '_'] = 'Kaley Cuoco'
    expected_res[Segment(8, 10), 'anything'] = 'Jim Parsons'

    mapping = {'Penny': 'Kaley Cuoco',
               'Sheldon': 'Jim Parsons',
               'Leonard': 'Johnny Galecki'}
    assert annotation.translate(mapping) == expected_res


def test_analyze(annotation):

    assert annotation.label_duration('Penny') == 4
    assert annotation.chart() == [('Penny', 4),
                                  ('Sheldon', 2),
                                  ('Leonard', 1.5)]
    assert annotation.argmax() == 'Penny'
