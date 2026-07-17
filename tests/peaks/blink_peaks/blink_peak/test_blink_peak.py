import numpy as np
import pytest

from spectrumlab.peaks.blink_peaks.blink_peak import BlinkPeak
from spectrumlab.types import Array


def test_blink_peak():
    peak = BlinkPeak(
        minima=(10, 14),
        maxima=(12,),
    )

    assert peak.n_numbers == 5
    np.testing.assert_array_equal(peak.index, np.array([0, 1, 2, 3, 4]))
    np.testing.assert_array_equal(peak.number, np.array([10, 11, 12, 13, 14]))

    assert peak.include(12) is True
    assert peak.include(9) is False
    assert "BlinkPeak(minima: (10, 14); maxima: (12,))" == repr(peak)


def test_blink_peak_with_except_edges():
    peak = BlinkPeak(
        minima=(10, 14),
        maxima=(12,),
        except_edges=True,
    )
    
    assert peak.n_numbers == 5
    np.testing.assert_array_equal(peak.index, np.array([1, 2, 3]))
    np.testing.assert_array_equal(peak.number, np.array([11, 12, 13]))

    assert peak.include(10) is False
    assert peak.include(14) is False


@pytest.mark.parametrize('minima, expected', [
    ((10, 14), np.array([0, 4])),
    ((9, 15), np.array([0, 1, 5, 6])),
])
def test_blink_peak_with_clipping(
    minima: tuple[int, int],
    expected: Array[int],
):
    peak = BlinkPeak(
        minima=minima,
        maxima=(11, 13),
        except_edges=False,
    )

    np.testing.assert_array_equal(peak.tail, expected)


@pytest.mark.parametrize('minima, expected', [
    ((10, 14), np.array([])),
    ((9, 15), np.array([1, 5])),
    ((8, 16), np.array([1, 2, 6, 7])),
])
def test_blink_peak_with_clipping_except_edges(
    minima: tuple[int, int],
    expected: Array[int],
):
    peak = BlinkPeak(
        minima=minima,
        maxima=(11, 13),
        except_edges=True,
    )

    np.testing.assert_array_equal(peak.tail, expected)
