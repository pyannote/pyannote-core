import pytest

from pyannote.core import Segment
from pyannote.core import SlidingWindow
from pyannote.core import SlidingWindowFeature
import numpy as np

@pytest.fixture
def features():
    data = np.arange(20).reshape(2, 10).T
    window = SlidingWindow(start=0., step=1., duration=2.)
    return SlidingWindowFeature(data, window)

@pytest.fixture
def one_feature():
    return SlidingWindowFeature(
        np.array([ [[0, 0, 0]], [[1, 1, 1]] ]),
        SlidingWindow()
    )

@pytest.fixture
def segment():
    return Segment(3.3, 6.7)

def test_crop_loose(features, segment):
    actual = features.crop(segment, mode='loose')
    expected = np.array([[2, 3, 4, 5, 6], [12, 13, 14, 15, 16]]).T
    np.testing.assert_array_equal(expected, actual)

def test_crop_strict(features, segment):
    actual = features.crop(segment, mode='strict')
    expected = np.array([[4, ], [14, ]]).T
    np.testing.assert_array_equal(expected, actual)

def test_crop_center(features, segment):
    actual = features.crop(segment, mode='center')
    expected = np.array([[2, 3, 4, 5, 6], [12, 13, 14, 15, 16]]).T
    np.testing.assert_array_equal(expected, actual)

def test_crop_fixed(features, segment):
    actual = features.crop(segment, mode='center', fixed=4.)
    expected = np.array([[2, 3, 4, 5], [12, 13, 14, 15]]).T
    np.testing.assert_array_equal(expected, actual)

def test_crop_out_of_bounds(features):
    segment = Segment(-6, -1)
    actual = features.crop(segment, mode='strict')
    expected = np.empty((0, 2))
    np.testing.assert_array_equal(expected, actual)

def test_crop_fixed_out_of_bounds(features):
    segment = Segment(-2, 6.7)
    actual = features.crop(segment, mode='center', fixed=8.7)
    expected = np.array([[0, 0, 0, 0, 1, 2, 3, 4, 5],
                         [10, 10, 10, 10, 11, 12, 13, 14, 15]]).T
    np.testing.assert_array_equal(expected, actual)

def test_repr_png(features):
    try:
        import matplotlib
        import IPython
    except ModuleNotFoundError:
        pytest.skip("notebook dependencies not available")
    expected = b'\x89PNG'
    actual = features._repr_png_()[:4]
    assert expected == actual

def test_repr_png_one_feature(one_feature):
    try:
        import matplotlib
        import IPython
    except ModuleNotFoundError:
        pytest.skip("notebook dependencies not available")
    expected = b'\x89PNG'
    actual = one_feature._repr_png_()[:4]
    assert expected == actual
