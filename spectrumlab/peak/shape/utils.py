from collections.abc import Sequence
from functools import partial
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize

from spectrumlab.grid import Grid
from spectrumlab.spectrum import Spectrum
from spectrumlab.utils import mse
from spectrumlab.typing import Array, Number

if TYPE_CHECKING:
    from spectrumlab.peak.analyte_peak import AnalytePeak
    from spectrumlab.peak.blink_peak import BlinkPeak
    from spectrumlab.peak.shape import PeakShape


# --------        approx        --------
def approx_peak_by_tail(peak: 'AnalytePeak', shape: 'PeakShape') -> float:
    """Approximate the analyte peak with selected shape on the tail"""

    # index
    index = peak.tail
    index = index[peak.mask[index]]

    # intensity
    x = peak.number[index]
    y = peak.value[index]
    y_hat = shape(x=x, **{'position': peak.position, 'intensity': 1})

    intensity = np.dot(y,y) / np.dot(y_hat,y)

    #
    return intensity


def approx_peak(peak: 'AnalytePeak', shape: 'PeakShape', delta: float = 1, by_tail: bool = False, show: bool = False) -> dict:
    """Approximate the analyte peak with selected shape."""

    def calculate_loss(params: Sequence[float], peak: 'AnalytePeak', shape: 'PeakShape', by_tail: bool = False) -> float:
        """Interface to calculate a loss of approximation of a analyte peak by any shape"""
        index = peak.tail if by_tail else peak.index
        x = peak.number[index]
        y = peak.value[index]

        return mse(
            y=y,
            y_hat=shape(x=x, **shape.approx_parse(params)),
        )

    #
    result = optimize.minimize(
        partial(calculate_loss, peak=peak, shape=shape, by_tail=by_tail),
        shape.approx_initial(peak=peak),
        method='SLSQP',
        bounds=shape.approx_bounds(peak=peak, delta=delta),
    )
    params = shape.approx_parse(result.x)

    # draw
    if show:
        x, y = peak.number, peak.value
        plt.plot(x, y, color='black', marker='s', markersize=.5, alpha=.2)

        x, y = peak.number, peak.value
        y[~peak.mask] = np.nan
        plt.plot(x, y, marker='s', markersize=.5)

        left, right = peak.minima
        x = np.linspace(left, right, 1000)
        y_hat = shape(x, **params)
        plt.plot(x, y_hat, color='red')

        x, y = peak.number, peak.value
        y_hat = shape(x, **params)
        plt.plot(x, y - y_hat, color='black', linestyle=':')

        plt.grid()
        plt.show()

    return params


# --------        grid        --------
def restore_grid_from_blinks(
    spectrum: Spectrum,
    blinks: Sequence['BlinkPeak'],
    offset: Array[Number] | None = None,
    scale: Array[float] | None = None,
    background: Array[float] | None = None,
    threshold: float = 0,
) -> Grid:
    """Get a grid from sequence of blinks from spectrum."""
    assert spectrum.n_times == 1, 'time resolved spectra are not supported!'

    #
    def _get_item(spectrum: Spectrum, blink: 'BlinkPeak', threshold: float) -> tuple[Array, Array]:
        lb, ub = blink.minima

        is_clipped = spectrum.clipped[lb:ub]
        is_snr_low = np.abs(spectrum.intensity[lb:ub]) / spectrum.deviation[lb:ub] < threshold
        mask = ~is_clipped & ~is_snr_low

        return spectrum.number[lb:ub][mask], spectrum.intensity[lb:ub][mask]

    items = tuple(
        _get_item(spectrum, blink=blink, threshold=threshold)
        for blink in blinks
    )

    #
    return _restore_grid_from_items(
        items=items,
        offset=offset,
        scale=scale,
        background=background,
    )


def restore_grid_from_frames(
    spectrum: Spectrum,
    offset: Array[Number] | None = None,
    scale: Array[float] | None = None,
    background: Array[float] | None = None,
    threshold: float = 0,
) -> Grid:
    """Get a grid from frames of spectra (for example, series of shifted on wavelength)."""
    # assert spectrum.n_times > 1, 'only time resolved spectra are supported!'

    #
    def _get_item(spectrum: Spectrum, t: int, threshold: float) -> tuple[Array, Array]:
        is_clipped = spectrum.clipped[t]
        is_snr_low = np.abs(spectrum.intensity[t]) / spectrum.deviation[t] < threshold
        mask = ~is_clipped & ~is_snr_low

        if spectrum.n_times == 1:
            return spectrum.number[mask], spectrum.intensity[mask]
        return spectrum.number[mask], spectrum.intensity[t, mask]

    items = tuple(
        _get_item(spectrum, t=t, threshold=threshold)
        for t in range(spectrum.n_times)
    )

    #
    return _restore_grid_from_items(
        items=items,
        offset=offset,
        scale=scale,
        background=background,
    )


# --------        private        --------
def _restore_grid_from_items(
    items: Sequence[tuple[Array, Array]],
    offset: Array[Number] | None = None,
    scale: Array[float] | None = None,
    background: Array[float] | None = None,
) -> Grid:
    """Get a grid from sequence of items."""
    n_times = len(items)

    if offset is None:
        offset = tuple(0 for _ in range(n_times))
    assert len(offset) == n_times, f'len of offset have to be equal of n_times: {n_times}'

    if scale is None:
        scale = tuple(1 for _ in range(n_times))
    assert len(scale) == n_times, f'len of scale have to be equal of n_times: {n_times}'

    if background is None:
        background = np.zeros(n_times)
    assert len(background) == n_times, f'len of background have to be equal of n_times: {n_times}'

    #
    _x, _y = [], []
    for t in range(n_times):
        x, y = items[t]

        _x.extend(x - offset[t])
        _y.extend((y - background[t]) / scale[t])
    _x, _y = np.array(_x).squeeze(), np.array(_y).squeeze()

    index = np.argsort(_x)

    #
    return Grid(
        x=_x[index],
        y=_y[index],
    )
