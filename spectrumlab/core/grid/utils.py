from abc import ABC
from collections.abc import Sequence
from functools import partial
from typing import Callable, TypeAlias

import numpy as np
from scipy import interpolate, optimize
import matplotlib.pyplot as plt

from spectrumlab.alias import Array
from spectrumlab.core.grid import Grid, T
from spectrumlab.core.approximate.scope import ScopeVariables
from spectrumlab.peak.shape.voigt_peak_shape import VoigtPeakShape
from spectrumlab.utils import mse


# --------        handlers        --------
class BaseHandler(ABC):

    def __init__(self, grid: Grid):
        self._grid = grid

    @property
    def grid(self) -> Grid:
        return self._grid

    @property
    def f(self) -> Callable[[Array[T]], Array[float]]:
        return self._f

    # --------        handlers        --------
    def show(self, bias: T | None = None):
        grid = self._grid

        if bias is None:
            bias = 1

        #
        fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)

        x, y = grid.x, grid.y
        plt.plot(
            x - bias, y,
            color='red', linestyle='none', marker='s', markersize=3,
            alpha=1,
        )

        x = grid.space()
        y_hat = self.f(x)
        plt.plot(
            x - bias, y_hat,
            color='black', linestyle='-', linewidth=1,
            alpha=1,
        )

        x, y = grid.x, grid.y
        y_hat = self.f(grid.x)
        plt.plot(
            x - bias, y - y_hat,
            color='black', linestyle='none', marker='s', markersize=0.5,
            alpha=1,
        )

        plt.xlabel(r'$number$' if grid.step == 1 else r'$x$ [$\mu m$]')
        plt.ylabel(r'$f(x)$')
        plt.grid(color='grey', linestyle=':')

        plt.show()

    # --------        private        --------
    def __call__(self, x: Array) -> Array:
        return self.f(x)


class LinearInterpolationHandler(BaseHandler):

    def __init__(self, grid: Grid, show: bool = False):
        super().__init__(grid=grid)

        #
        self._f = interpolate.interp1d(
            grid.x, grid.y,
            kind='linear',
            bounds_error=False,
            fill_value=0,
        )

        # show
        if show:
            self.show()


class VoigtPeakShapeHandler(BaseHandler):

    def __init__(self, grid: Grid, show: bool = False):
        super().__init__(grid=grid)

        # shape
        shape = VoigtPeakShape.from_grid(grid=grid)

        # scope
        def _loss(params: Sequence[float], grid: Grid, shape: VoigtPeakShape) -> float:
            scope_variables = ScopeVariables(grid, *params)

            y = grid.y
            y_hat = shape(x=grid.x, **scope_variables)

            return mse(y, y_hat)

        variables = ScopeVariables(grid=grid)
        res = optimize.minimize(
            partial(_loss, grid=grid, shape=shape),
            variables.initial,
            method='SLSQP',
            bounds=variables.bounds,
        )
        assert res['success'], 'Optimization is not succeeded!'

        scope_variables = ScopeVariables(grid, *res['x'])

        # f
        self._f = partial(shape, **scope_variables)

        # show
        if show:
            self.show(bias=scope_variables['position'])


Handler: TypeAlias = LinearInterpolationHandler | VoigtPeakShapeHandler


# --------        estimators        --------
def estimate_bias(grid: Grid, handler: Handler | None = None, verbose: bool = False, show: bool = False) -> T:
    '''Estimate a bias of the `grid`.

    Params:
        handler: Handler | None = None - approximate the grid by smooth function (peak's voigt shape)
    
    '''
    if handler is None:
        handler = LinearInterpolationHandler(grid=grid)

    # bias
    def _loss(x: T, handler: Callable[[T], float], step: T) -> float:
        return (handler(x - step/2) - handler(x + step/2))**2

    bias = optimize.minimize(
        partial(_loss, handler=handler, step=grid.step),
        x0=grid.x[np.argmax(grid.y)],  # FIXME: change to maximum of `handler`!
    )['x'][0]

    # verbose
    if verbose:
        content = '\n'.join([
            'bias: {value:.4f} {units}'.format(
                value=bias,
                units='' if grid.step == 1 is None else r'[micro]',
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
                units='' if grid.step == 1 is None else r'[micro]',
            ),
        ])
        plt.text(
            0.05, 0.95,
            content,
            transform=ax.transAxes,
            ha='left', va='top',
        )

        plt.xlabel(r'$number$' if grid.step == 1 else r'$x$ [$\mu m$]')
        plt.ylabel(r'$f(x)$')
        plt.grid(color='grey', linestyle=':')

        plt.show()

    #
    return bias


def estimate_fwhm(grid: Grid, handler: Handler | None = None, bias: T = 0, lim: float = 1, verbose: bool = False, show: bool = False) -> T:
    """Estimate a full width at half maximum (FWHM) of the `grid`.

    Params:
        handler: 
        lim: float - the lowest lim of full width at half maximum 

    A grid should be centered!
    """
    if handler is None:
        handler = LinearInterpolationHandler(grid=grid)

    x0 = bias
    rx = grid.step*lim/2

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
                units='' if grid.step == 1 else r'[micro]',
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
                units='' if grid.step == 1 else r'[$\mu m$]',
            ),
        ])
        plt.text(
            0.05, 0.95,
            content,
            transform=ax.transAxes,
            ha='left', va='top',
        )

        plt.xlabel(r'$number$' if grid.step == 1 else r'$x$ [$\mu m$]')
        plt.ylabel(r'$f(x)$')
        plt.grid(color='grey', linestyle=':')

        plt.show()

    # 
    return fwhm
