from functools import partial
from typing import Callable, TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize

from spectrumlab.grids.filter import GridFilterABC, LinearInterpolationGridFilter
from spectrumlab.types import T

if TYPE_CHECKING:
    from spectrumlab.grids import Grid


def estimate_bias(
    grid: 'Grid',
    pitch: T,
    filter: GridFilterABC | None = None,
    verbose: bool = False,
    show: bool = False,
) -> T:
    """Estimate a bias of the `grid`."""
    filter = filter or LinearInterpolationGridFilter(grid=grid)

    def calculate_loss(x: T, filter: Callable[[T], float], pitch: T) -> float:
        return (filter(x - pitch/2) - filter(x + pitch/2))**2

    bias = optimize.minimize(
        partial(calculate_loss, filter=filter, pitch=pitch),
        x0=grid.x[np.argmax(grid.y)],  # FIXME: change to maximum of `filter`!
    )['x'][0]

    if verbose:
        content = '\n'.join([
            'bias: {value:.4f} {units}'.format(
                value=bias,
                units=grid.xunits,
            ),
        ])
        print(content)

    if show:
        fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)

        x = bias
        plt.axvline(
            x,
            color='red', linestyle='--', linewidth=1,
            alpha=1,
        )

        x, y = grid.x, grid.y
        plt.plot(
            x, y,
            color='red', linestyle='none', marker='s', markersize=3,
            alpha=1,
        )

        x = grid.space()
        f_hat = filter(x)
        plt.plot(
            x, f_hat,
            color='black', linestyle='-', linewidth=1,
            alpha=1,
        )

        content = '\n'.join([
            'bias: {value:.4f} {units}'.format(
                value=bias,
                units=grid.xunits,
            ),
        ])
        plt.text(
            0.05, 0.95,
            content,
            transform=ax.transAxes,
            ha='left', va='top',
        )

        plt.xlabel(grid.xlabel)
        plt.ylabel(r'$f(x)$')
        plt.grid(color='grey', linestyle=':')

        plt.show()

    return bias


def estimate_fwhm(
    grid: 'Grid',
    pitch: T,  # minimum `hwhm`
    position: T = 0,
    filter: GridFilterABC | None = None,
    verbose: bool = False,
    show: bool = False,
) -> T:
    """Estimate a full width at half maximum (FWHM) of the `grid`."""
    filter = filter or LinearInterpolationGridFilter(grid=grid)

    def calculate_loss(x: T, filter: Callable[[T], float], y: float) -> float:
        y_hat = filter(x)
        return (y_hat - y)**2

    mask = grid.x < position
    res = optimize.minimize(
        partial(calculate_loss, filter=filter, y=filter(position)/2),
        grid.x[mask][np.argmin(np.abs(filter(grid.x[mask]) - filter(position)/2))],
        bounds=[
            (np.min(grid.x), position - pitch/2),
        ],
        tol=1e-10,
    )
    # assert res['success'], 'Optimization is not success!'
    lb = res['x'].item()

    mask = grid.x > position
    res = optimize.minimize(
        partial(calculate_loss, filter=filter, y=filter(position)/2),
        grid.x[mask][np.argmin(np.abs(filter(grid.x[mask]) - filter(position)/2))],
        bounds=[
            (position + pitch/2, np.max(grid.x)),
        ],
        tol=1e-10,
    )
    # assert res['success'], 'Optimization is not success!'
    rb = res['x'].item()

    fwhm = rb - lb

    if verbose:
        content = '\n'.join([
            'FWHM: {value:.4f} {units}'.format(
                value=fwhm,
                units=grid.xunits,
            ),
        ])
        print(content)

    if show:
        fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)

        plt.axvline(
            lb,
            color='grey', linestyle='--', linewidth=1,
            alpha=1,
        )
        plt.axvline(
            rb,
            color='grey', linestyle='--', linewidth=1,
            alpha=1,
        )
        plt.axhline(
            filter(position)/2,
            color='grey', linestyle='--', linewidth=1,
            alpha=1,
        )

        x, y = grid.x, grid.y
        plt.plot(
            x, y,
            color='red', linestyle='none', marker='s', markersize=3,
            alpha=1,
        )

        x = grid.space()
        f_hat = filter(x)
        plt.plot(
            x, f_hat,
            color='black', linestyle='-', linewidth=1,
            alpha=1,
        )

        x = np.array([lb, rb])
        y = filter(x)
        plt.plot(
            x, y,
            color='red', linestyle='--', linewidth=1,
            alpha=1,
        )

        content = '\n'.join([
            'FWHM: {value:.2f} {units}'.format(
                value=fwhm,
                units=grid.xunits,
            ),
        ])
        plt.text(
            0.05, 0.95,
            content,
            transform=ax.transAxes,
            ha='left', va='top',
        )

        plt.xlabel(grid.xlabel)
        plt.ylabel(r'$f(x)$')
        plt.grid(color='grey', linestyle=':')

        plt.show()

    return fwhm


# if __name__ == '__main__':

#     from spectrumlab.detectors import Detector
#     from spectrumlab.grids import Grid
#     from spectrumlab.types import MicroMeter, Number, T
#     from spectrumlab_emulations.aperture import MeasuredApertureShape

#     for detector in [Detector.BLPP369M1, Detector.BLPP2000, Detector.BLPP4000]:
#         rx = 5
#         dx = 1e-2
#         x = np.linspace(-rx, +rx, 2*int(rx/dx))
#         f = MeasuredApertureShape.from_datasheet(detector)

#         grid = Grid(
#             x=x, y=f(x, 0),
#             units=Number,
#         ).rescale(
#             1/detector.pitch, units=MicroMeter,
#         )

#         fwhm = estimate_fwhm(
#             grid=grid,
#             pitch=detector.pitch,
#             verbose=True,
#             show=True,
#         )
