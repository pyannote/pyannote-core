import numpy as np
import pytest

from pyannote.core import Segment, SlidingWindow, SlidingWindowFeature


@pytest.fixture
def features():
    data = np.arange(20).reshape(2, 10).T
    window = SlidingWindow(start=0.0, step=1.0, duration=2.0)
    return SlidingWindowFeature(data, window)


@pytest.fixture
def one_feature():
    return SlidingWindowFeature(np.array([[[0, 0, 0]], [[1, 1, 1]]]), SlidingWindow())


@pytest.fixture
def segment():
    return Segment(3.3, 6.7)


def test_crop_loose(features, segment):
    actual = features.crop(segment, mode="loose")
    expected = np.array([[2, 3, 4, 5, 6], [12, 13, 14, 15, 16]]).T
    np.testing.assert_array_equal(expected, actual)


def test_crop_strict(features, segment):
    actual = features.crop(segment, mode="strict")
    expected = np.array(
        [
            [
                4,
            ],
            [
                14,
            ],
        ]
    ).T
    np.testing.assert_array_equal(expected, actual)


def test_crop_center(features, segment):
    actual = features.crop(segment, mode="center")
    expected = np.array([[2, 3, 4, 5, 6], [12, 13, 14, 15, 16]]).T
    np.testing.assert_array_equal(expected, actual)


def test_crop_fixed(features, segment):
    actual = features.crop(segment, mode="center", fixed=4.0)
    expected = np.array([[2, 3, 4, 5], [12, 13, 14, 15]]).T
    np.testing.assert_array_equal(expected, actual)


def test_crop_out_of_bounds(features):
    segment = Segment(-6, -1)
    actual = features.crop(segment, mode="strict")
    expected = np.empty((0, 2))
    np.testing.assert_array_equal(expected, actual)


def test_crop_fixed_out_of_bounds(features):
    segment = Segment(-2, 6.7)
    actual = features.crop(segment, mode="center", fixed=8.7)
    expected = np.array(
        [[0, 0, 0, 0, 1, 2, 3, 4, 5], [10, 10, 10, 10, 11, 12, 13, 14, 15]]
    ).T
    np.testing.assert_array_equal(expected, actual)


def test_repr_png(features):
    pytest.importorskip("IPython")
    pytest.importorskip("matplotlib")
    expected = b"\x89PNG"
    actual = features._repr_png_()[:4]
    assert expected == actual


def test_repr_png_one_feature(one_feature):
    pytest.importorskip("IPython")
    pytest.importorskip("matplotlib")
    expected = b"\x89PNG"
    actual = one_feature._repr_png_()[:4]
    assert expected == actual


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_array_dtype(features, dtype):
    actual = np.asarray(features, dtype=dtype)
    np.testing.assert_array_equal(actual, features.data.astype(dtype))
    assert actual.dtype == dtype


@pytest.mark.parametrize("copy", [False, True, None])
def test_array_copy(features, copy):
    actual = np.array(features, copy=copy)
    np.testing.assert_array_equal(actual, features.data)
    assert np.shares_memory(actual, features.data) == (copy is not True)


def test_array_copy_false_dtype_change(features):
    with pytest.raises(ValueError):
        np.array(features, dtype=np.float32, copy=False)
