from collections.abc import Sequence
from functools import partial
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize

from spectrumlab.utils import mse

if TYPE_CHECKING:
    from spectrumlab.peaks.analyte_peaks.analyte_peak import AnalytePeak
    from spectrumlab.peaks.analyte_peaks.shapes import PeakShape


def approx_peak_by_tail(peak: 'AnalytePeak', shape: 'PeakShape') -> float:
    """Approximate the analyte peak with selected shape on the tail"""

    index = peak.tail
    index = index[peak.mask[index]]

    x = peak.number[index]
    y = peak.value[index]
    y_hat = shape(x=x, **{'position': peak.position, 'intensity': 1})

    intensity = np.dot(y, y) / np.dot(y_hat, y)
    return intensity


def approx_peak(
    peak: 'AnalytePeak',
    shape: 'PeakShape',
    delta: float = 1,
    by_tail: bool = False,
    verbose: bool = False,
    show: bool = False,
) -> dict:
    """Approximate the analyte peak with selected shape."""

    def calculate_loss(params: Sequence[float], peak: 'AnalytePeak', shape: 'PeakShape', by_tail: bool) -> float:
        """Interface to calculate a loss of approximation of a analyte peak by any shape"""
        index = peak.tail if by_tail else peak.index
        index = index[peak.mask[index]]
        x = peak.number[index]
        y = peak.value[index]

        return mse(
            y=y,
            y_hat=shape(x=x, **shape.parse_params(params)),
        )

    result = optimize.minimize(
        partial(calculate_loss, peak=peak, shape=shape, by_tail=by_tail),
        shape.approx_initial(peak=peak),
        method='SLSQP',
        bounds=shape.approx_bounds(peak=peak, delta=delta),
    )
    params = shape.parse_params(result.x)

    if verbose:
        print('Initial:', shape.parse_params(shape.approx_initial(peak=peak)))
        print('Bounds:', shape.parse_params(shape.approx_bounds(peak=peak, delta=delta)))
        print('Result:', shape.parse_params(result.x))

    if show:

        x, y = peak.number, peak.value
        plt.step(
            x, y,
            where='mid',
            color='black',
        )

        left, right = peak.minima
        x = np.linspace(left, right, 1000)
        y_hat = shape(x, **params)
        plt.plot(
            x, y_hat,
            color='#9467bd',
        )

        index = peak.mask
        plt.plot(
            peak.number[index], peak.value[index],
            color='red', linestyle='none', marker='s', markersize=3,
        )

        # x, y = peak.number, peak.value
        # y_hat = shape(x, **params)
        # plt.step(
        #     x, y - y_hat,
        #     where='mid',
        #     color='black', linestyle=':',
        # )

        plt.grid(color='grey', linestyle=':')
        plt.show()

    return params
