import pytest

from spectrumlab.peaks.blink_peaks.draft_blinks.draft_blinks import find_maxima
from spectrumlab.types import Array


@pytest.mark.parametrize(('array', 'expected'), [
    ([1, 10, 1], (1,)),
    ([1, 10, 1, 10, 1], (1, 3)),
])
def test_find_maxima(array: Array, expected: tuple[int, ...]):

    assert find_maxima(array) == expected


@pytest.mark.parametrize(('array', 'expected'), [
    ([3, 2, 1], (0,)),
    ([1, 2, 3], (2,)),
    ([2, 1, 2], (0, 2)),
])
def test_find_maxima_with_peaks_at_edges(array: Array, expected: tuple[int, ...]):

    assert find_maxima(array) == expected


@pytest.mark.parametrize(('array', 'expected'), [
    ([1, 2, 2, 1], (1, 2)),
    ([1, 2, 2, 2, 1], (1, 3)),
])
def test_find_maxima_with_clipped(array: Array, expected: tuple[int, ...]):

    assert find_maxima(array) == expected


@pytest.mark.parametrize(('array', 'expected'), [
    ([2, 2, 2], ()),
])
def test_find_maxima_without_peaks(array: Array, expected: tuple[int, ...]):

    assert find_maxima(array) == expected


@pytest.mark.parametrize('array', [
    (),
    (1,),
])
def test_find_maxima_with_too_short_array(array: Array):

    with pytest.raises(ValueError):
        find_maxima(array)
