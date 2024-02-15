from enum import Enum, auto
from functools import partial
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate, interpolate, optimize

from spectrumlab.alias import Array
from spectrumlab.core.grid import Grid, T
from spectrumlab.core.grid.handler import Handler, LinearInterpolationHandler


# --------        handlers        --------
class InterpolationKind(Enum):
    NEAREST = auto()
    LINEAR = auto()


def interpolate_grid(grid: Grid, kind: InterpolationKind) -> Callable[[Array[T]], Array[float]]:
    """Interpolate the grid."""

    return interpolate.interp1d(
        grid.x, grid.y,
        kind={
            InterpolationKind.NEAREST: 'nearest',
            InterpolationKind.LINEAR: 'linear',
        }.get(kind),
        bounds_error=False,
        fill_value=0,
    )


def integrate_grid(grid: Grid, position: float, interval: float, kind: InterpolationKind = InterpolationKind.LINEAR) -> float:
    """Integrate the grid in given `position` and `interval`."""

    return integrate.quad(
        interpolate_grid(grid, kind=kind),
        a=position - interval/2,
        b=position + interval/2,
    )[0]


# --------        estimators        --------
def estimate_bias(grid: Grid, pitch: T, handler: Handler | None = None, verbose: bool = False, show: bool = False) -> T:
    """Estimate a bias of the `grid`."""
    handler = handler or LinearInterpolationHandler(grid=grid)

    # bias
    def calculate_loss(x: T, handler: Callable[[T], float], pitch: T) -> float:
        return (handler(x - pitch/2) - handler(x + pitch/2))**2

    bias = optimize.minimize(
        partial(calculate_loss, handler=handler, pitch=pitch),
        x0=grid.x[np.argmax(grid.y)],  # FIXME: change to maximum of `handler`!
    )['x'][0]

    # verbose
    if verbose:
        content = '\n'.join([
            'bias: {value:.4f} {units}'.format(
                value=bias,
                units=grid.xunits,
            ),
        ])
        print(content)

    # show
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
        f_hat = handler(x)
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

    #
    return bias


def estimate_fwhm(grid: Grid, pitch: T, position: T = 0, handler: Handler | None = None, verbose: bool = False, show: bool = False) -> T:
    """Estimate a full width at half maximum (FWHM) of the `grid`."""
    handler = handler or LinearInterpolationHandler(grid=grid)

    # fwhm
    def calculate_loss(x: T, handler: Callable[[T], float], y: float) -> float:
        y_hat = handler(x)
        return (y_hat - y)**2

    res = optimize.minimize(
        partial(calculate_loss, handler=handler, y=handler(position)/2),
        position - pitch/2,
        bounds=[
            (-np.inf, position - pitch/2),
        ],
        tol=1e-10,
    )
    assert res['success'], 'Optimization is not success!'
    lb = res['x'].item()

    res = optimize.minimize(
        partial(calculate_loss, handler=handler, y=handler(position)/2),
        position + pitch/2,
        bounds=[
            (position + pitch/2, +np.inf),
        ],
        tol=1e-10,
    )
    assert res['success'], 'Optimization is not success!'
    rb = res['x'].item()

    fwhm = rb - lb

    # verbose
    if verbose:
        content = '\n'.join([
            'FWHM: {value:.4f} {units}'.format(
                value=fwhm,
                units=grid.xunits,
            ),
        ])
        print(content)

    # show
    if show:
        fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)

        plt.axvline(
            lb,
            color='grey', linestyle=':', linewidth=1,
            alpha=1,
        )
        plt.axvline(
            rb,
            color='grey', linestyle=':', linewidth=1,
            alpha=1,
        )
        plt.axhline(
            handler(position)/2,
            color='grey', linestyle=':', linewidth=1,
            alpha=1,
        )

        x, y = grid.x, grid.y
        plt.plot(
            x, y,
            color='red', linestyle='none', marker='s', markersize=3,
            alpha=1,
        )

        x = grid.space()
        f_hat = handler(x)
        plt.plot(
            x, f_hat,
            color='black', linestyle='-', linewidth=1,
            alpha=1,
        )

        x = np.array([lb, rb])
        y = handler(x)
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

    #
    return fwhm


if __name__ == '__main__':
    from spectrumlab.alias import Array, MicroMeter, Number
    from spectrumlab.emulation.aperture import MeasuredApertureShape
    from spectrumlab.emulation.detector import Detector

    for detector in [Detector.BLPP369M1, Detector.BLPP2000, Detector.BLPP4000]:
        rx = 5
        dx = 1e-2
        x: Array[float] = np.linspace(-rx, +rx, 2*int(rx/dx))
        f = MeasuredApertureShape.from_datasheet(detector)

        grid = Grid(
            x=x, y=f(x, 0),
            units=Number,
        ).rescale(
            1/detector.pitch, units=MicroMeter,
        )

        fwhm = estimate_fwhm(
            grid=grid,
            pitch=detector.pitch,
            verbose=True,
            show=True,
        )
