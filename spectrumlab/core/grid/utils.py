from functools import partial
from typing import Callable

import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

from spectrumlab.core.grid import Grid, T
from spectrumlab.core.grid.handler import Handler, LinearInterpolationHandler


# --------        estimators        --------
def estimate_bias(grid: Grid, pitch: T, handler: Handler | None = None, verbose: bool = False, show: bool = False) -> T:
    '''Estimate a bias of the `grid`.'''
    handler = handler or LinearInterpolationHandler(grid=grid)

    # bias
    def _loss(x: T, handler: Callable[[T], float], pitch: T) -> float:
        return (handler(x - pitch/2) - handler(x + pitch/2))**2

    bias = optimize.minimize(
        partial(_loss, handler=handler, pitch=pitch),
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


def estimate_fwhm(grid: Grid, pitch: T, bias: T = 0, handler: Handler | None = None, verbose: bool = False, show: bool = False) -> T:
    """Estimate a full width at half maximum (FWHM) of the `grid`."""
    handler = handler or LinearInterpolationHandler(grid=grid)
    x0 = bias
    rx = pitch/2

    # fwhm
    def _loss(x: T, handler: Callable[[T], float], y: float) -> float:
        y_hat = handler(x)
        return (y_hat - y)**2

    res = optimize.minimize(
        partial(_loss, handler=handler, y=handler(x0)/2),
        x0=x0-rx,
        bounds=[
            (-np.inf, x0-rx),
        ],
    )
    assert res['success'], 'Optimization is not success!'
    lb = res['x'].item()

    res = optimize.minimize(
        partial(_loss, handler=handler, y=handler(x0)/2),
        x0=x0+rx,
        bounds=[
            (x0+rx, +np.inf),
        ],
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

        x = x0-rx
        plt.axvline(
            x,
            color='grey', linestyle=':', linewidth=1,
            alpha=1,
        )
        x = x0 + rx
        plt.axvline(
            x,
            color='grey', linestyle=':', linewidth=1,
            alpha=1,
        )
        y = handler(x0)/2
        plt.axhline(
            y,
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
