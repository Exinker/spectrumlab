from typing import Iterator, Sequence

import numpy as np
from scipy import signal

from spectrumlab.peaks.blink_peaks import BlinkPeak
from spectrumlab.peaks.blink_peaks.draft_blinks import DRAFT_BLINKS_CONFIG, DraftBlinksConfig
from spectrumlab.spectra import Spectrum
from spectrumlab.types import Array, Number, R


def draft_blinks(
    spectrum: Spectrum,
    config: DraftBlinksConfig | None = None,
) -> tuple[BlinkPeak, ...]:
    """Draft blink peaks from the spectrum.

    Author: Vaschenko Pavel
     Email: vaschenko@vmk.ru
      Date: 2016.04.09
    """
    config = config or DRAFT_BLINKS_CONFIG

    # find pairs of local minima for each maximum
    maxima = find_maxima(spectrum.intensity)
    minima = find_minima(spectrum.intensity)
    pairs = find_pairs(
        maxima=maxima,
        minima=minima,
    )

    # find width
    width, *_ = signal.peak_widths(spectrum.intensity, maxima)

    # draft peaks
    peaks = []
    for i, (maximum, pair) in enumerate(zip(maxima, pairs)):
        left, right = pair  # left and right index of peak

        # check peaks's width
        if config.except_wide_peak:
            if width[i] > config.width_max:
                continue

        # check n_counts
        n_counts = right - left + 1

        if n_counts < config.n_counts_min:
            continue

        if n_counts > config.n_counts_max:
            continue

        # check peaks's amplitude
        _amplitude = spectrum.intensity[maximum] - (spectrum.intensity[left] + spectrum.intensity[right])/2  # noqa: E501 - от среднего значения на границах до максимума
        _deviation = (spectrum.deviation[maximum]**2 + .25*spectrum.deviation[left]**2 + .25*spectrum.deviation[right]**2)**0.5  # noqa: E501

        if np.isnan(_amplitude):
            continue

        if _amplitude < config.amplitude_min:
            continue

        if _amplitude < config.noise_level * _deviation:
            continue

        # check clipped counts
        if config.except_clipped_peak:
            if any(spectrum.clipped[left:right+1]):
                continue

        # check peaks's slope
        if config.except_sloped_peak:
            _slope = abs(spectrum.intensity[left] - spectrum.intensity[right]) / _amplitude

            if _slope > config.slope_max:
                continue

        # gather peak
        peak = BlinkPeak(
            minima=(left, right),
            maxima=(maximum,),

            except_edges=config.except_edges,
        )
        peaks.append(peak)

    unique_peaks = {}
    for peak in peaks:
        if peak.minima not in unique_peaks:
            unique_peaks[peak.minima] = peak

        else:
            unique_peaks[peak.minima].maxima += peak.maxima

    return tuple(unique_peaks.values())


def find_minima(__value: Array[R]) -> tuple[Number, ...]:
    """Find local minima index."""
    n_values = len(__value)
    extrema = []

    # add the first index
    i = 0
    extrema.append(i)

    # add the middle index
    for i in range(1, n_values-1):
        conditions = (
            __value[i-1] > __value[i] and __value[i] <= __value[i+1],
            __value[i-1] >= __value[i] and __value[i] < __value[i+1],
        )
        if any(conditions):
            extrema.append(i)

    # add the last index
    i = n_values-1
    extrema.append(i)

    return tuple(extrema)


def find_maxima(__value: Array[R]) -> tuple[Number, ...]:
    """Find local maxima index."""
    extrema = []

    n_values = len(__value)
    if n_values < 2:
        raise ValueError('Array must contain at least 2 elements')

    # add the first index
    i = 0
    if __value[i] > __value[i+1]:
        extrema.append(i)

    # add the middle index
    for i in range(1, n_values-1):
        conditions = (
            __value[i-1] < __value[i] and __value[i] >= __value[i+1],
            __value[i-1] <= __value[i] and __value[i] > __value[i+1],
        )
        if any(conditions):
            extrema.append(i)

    # add the last index
    i = n_values-1
    if __value[i-1] < __value[i]:
        extrema.append(i)

    return tuple(extrema)


def get_pairwise(values: Sequence[Number]) -> Iterator:
    """Get sequence by pairwise.

    Example:
        a, b, c, d, ... -> ((a, b), (b, c), (c, d), ...)
    """

    for a, b in zip(values, values[1:]):
        yield a, b


def find_pairs(maxima: Sequence[Number], minima: Sequence[Number]) -> tuple[tuple[Number, Number], ...]:
    """Find pairs (from a sequense of minima) for each of maxima."""
    pairs = list(get_pairwise(minima))

    edges = []
    for maximum in maxima:

        for left, right in pairs:
            if left <= maximum <= right:
                edge = left, right
                edges.append(edge)

                break

    return tuple(edges)
