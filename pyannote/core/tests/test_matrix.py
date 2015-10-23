import pytest

from pyannote.core.matrix import LabelMatrix
import numpy as np

@pytest.fixture
def matrix():
    matrix = LabelMatrix(data=[[1,2,3],[4,5,6]], rows=['A', 'B'], columns=['a', 'b', 'c'])
    return matrix


def test_itervalues(matrix):

    assert list(matrix.itervalues()) == [('A', 'a', 1),
                                         ('A', 'b', 2),
                                         ('A', 'c', 3),
                                         ('B', 'a', 4),
                                         ('B', 'b', 5),
                                         ('B', 'c', 6)]

def test_set(matrix):
    matrix['C', 'a'] = 7
    assert list(matrix.itervalues()) == [('A', 'a', 1),
                                         ('A', 'b', 2),
                                         ('A', 'c', 3),
                                         ('B', 'a', 4),
                                         ('B', 'b', 5),
                                         ('B', 'c', 6),
                                         ('C', 'a', 7)]

    assert np.isnan(matrix['C', 'b'])

def test_get(matrix):
    assert matrix['B', 'a'] == 4

def test_get_rows(matrix):
    assert matrix.get_rows() == ['A', 'B']

def test_get_columns(matrix):
    assert matrix.get_columns() == ['a', 'b', 'c']

def test_shape(matrix):
    assert matrix.shape == (2, 3)

def test_argmax(matrix):

    assert matrix.argmax(axis=0) == {'a': 'B', 'b': 'B', 'c': 'B'}
    assert matrix.argmax(axis=1) == {'A': 'c', 'B': 'c'}
    assert matrix.argmax() == {'B': 'c'}

def test_neg(matrix):

    assert list((-matrix).itervalues()) == [('A', 'a', -1),
                                            ('A', 'b', -2),
                                            ('A', 'c', -3),
                                            ('B', 'a', -4),
                                            ('B', 'b', -5),
                                            ('B', 'c', -6)]

def test_argmin(matrix):

    assert matrix.argmin(axis=0) == {'a': 'A', 'b': 'A', 'c': 'A'}
    assert matrix.argmin(axis=1) == {'A': 'a', 'B': 'a'}
    assert matrix.argmin() == {'A': 'a'}

def test_transpose(matrix):

    assert list(matrix.T.itervalues()) == [('a', 'A', 1),
                                           ('a', 'B', 4),
                                           ('b', 'A', 2),
                                           ('b', 'B', 5),
                                           ('c', 'A', 3),
                                           ('c', 'B', 6)]

def test_remove_column(matrix):

    matrix.remove_column('b')
    assert list(matrix.itervalues()) == [('A', 'a', 1),
                                         ('A', 'c', 3),
                                         ('B', 'a', 4),
                                         ('B', 'c', 6)]

def test_remove_row(matrix):
    matrix.remove_row('B')
    assert list(matrix.itervalues()) == [('A', 'a', 1),
                                         ('A', 'b', 2),
                                         ('A', 'c', 3)]

def test_copy(matrix):
    pass


def test_subset(matrix):
    subset = matrix.subset(rows=set('A'))
    assert list(subset.itervalues()) == [('A', 'a', 1),
                                         ('A', 'b', 2),
                                         ('A', 'c', 3)]

    subset = matrix.subset(columns=set('ac'))
    assert list(subset.itervalues()) == [('A', 'a', 1),
                                         ('A', 'c', 3),
                                         ('B', 'a', 4),
                                         ('B', 'c', 6)]

def test_greaterThan(matrix):
    assert list((matrix > 3).itervalues()) == [('A', 'a', False),
                                               ('A', 'b', False),
                                               ('A', 'c', False),
                                               ('B', 'a', True),
                                               ('B', 'b', True),
                                               ('B', 'c', True)]
