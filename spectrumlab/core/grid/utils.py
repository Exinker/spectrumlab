from abc import ABC
from collections.abc import Sequence
from functools import partial
from typing import Callable, TypeAlias, overload

import numpy as np
from scipy import interpolate, optimize
import matplotlib.pyplot as plt

from spectrumlab.alias import Array, MicroMeter
from spectrumlab.core.grid import Grid, T
from spectrumlab.peak.shape.scope import ScopeVariables
from spectrumlab.peak.shape.voight_peak_shape import VoightPeakShape
from spectrumlab.utils import mse


@overload
def to_scale(__value: T, scale: MicroMeter) -> MicroMeter: ...
@overload
def to_scale(__value: T, scale: None) -> T: ...
def to_scale(__value, scale):
    if scale is None:
        return __value
    
    return scale * __value


# --------        handlers        --------
class BaseHandler(ABC):

    def __init__(self, grid: Grid, scale: MicroMeter | None = None):
        self._grid = grid
        self._scale = scale

    @property
    def f(self) -> Callable[[Array[T]], Array[float]]:
        return self._f

    # --------        handlers        --------
    def show(self, bias: T | None = None):
        grid = self._grid
        scale = self._scale
        f = self.f

        if bias is None:
            bias = 1

        #
        fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)

        x, y = grid.x, grid.y
        plt.plot(
            to_scale(x - bias, scale=scale), y,
            color='red', linestyle='none', marker='s', markersize=3,
            alpha=1,
        )

        x = grid.space()
        y_hat = f(x)
        plt.plot(
            to_scale(x - bias, scale=scale), y_hat,
            color='black', linestyle='-', linewidth=1,
            alpha=1,
        )

        x, y = grid.x, grid.y
        y_hat = f(grid.x)
        plt.plot(
            to_scale(x - bias, scale=scale), y - y_hat,
            color='black', linestyle='none', marker='s', markersize=0.5,
            alpha=1,
        )

        lim = 4
        plt.xlim([-to_scale(lim, scale=scale), +to_scale(lim, scale=scale)])
        plt.xlabel(r'$number$' if scale is None else r'$x$ [$\mu m$]')
        plt.ylabel(r'$I$ [$\%$]')
        plt.grid(color='grey', linestyle=':')

        plt.show()

    # --------        private        --------
    def __call__(self, x: Array) -> Array:
        return self.f(x)


class LinearInterpolationHandler(BaseHandler):

    def __init__(self, grid: Grid, scale: MicroMeter | None = None, show: bool = False):
        super().__init__(grid=grid, scale=scale)

        #
        self._f = interpolate.interp1d(
            grid.x, grid.y,
            kind='linear',
            bounds_error=False,
            fill_value=0,
        )

        # show
        if show:
            self.show(scale=scale)


class VoightPeakShapeHandler(BaseHandler):

    def __init__(self, grid: Grid, scale: MicroMeter | None = None, show: bool = False):
        super().__init__(grid=grid, scale=scale)

        # shape
        shape = VoightPeakShape.from_grid(grid=grid)

        # scope
        def _loss(params: Sequence[float], grid: Grid, shape: VoightPeakShape) -> float:
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


Handler: TypeAlias = LinearInterpolationHandler | VoightPeakShapeHandler


# --------        estimators        --------
def estimate_bias(grid: Grid, handler: Handler | None = None, scale: MicroMeter | None = None, verbose: bool = False, show: bool = False) -> T:
    '''Estimate a bias of the `grid`.

    Params:
        handler: Handler | None = None - approximate the grid by smooth function (peak's voight shape)
    
    '''
    if handler is None:
        handler = LinearInterpolationHandler(grid=grid)

    # bias
    def _loss(x: T, handler: Callable[[T], float]) -> float:
        return (handler(x - 0.5) - handler(x + 0.5))**2

    bias = optimize.minimize(
        partial(_loss, handler=handler),
        x0=grid.x[np.argmax(grid.y)],  # FIXME: change to maximum of `handler`!
    )['x'][0]

    # verbose
    if verbose:
        content = '\n'.join([
            'bias: {value:.4f} {units}'.format(
                value=to_scale(bias, scale=scale),
                units='' if scale is None else r'[micro]',
            ),
        ])
        print(content)

    # show
    if show:
        fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)

        x = bias
        plt.axvline(
            to_scale(x, scale=scale),
            color='red', linestyle='--', linewidth=1,
            alpha=1,
        )

        x, y = grid.x, grid.y
        plt.plot(
            to_scale(x, scale=scale), y,
            color='red', linestyle='none', marker='s', markersize=3,
            alpha=1,
        )

        x = grid.space()
        f_hat = handler(x)
        plt.plot(
            to_scale(x, scale=scale), f_hat,
            color='black', linestyle='-', linewidth=1,
            alpha=1,
        )

        content = '\n'.join([
            'bias: {value:.4f} {units}'.format(
                value=to_scale(bias, scale=scale),
                units='' if scale is None else r'[$\mu m$]',
            ),
        ])
        plt.text(
            0.05, 0.95,
            content,
            transform=ax.transAxes,
            ha='left', va='top',
        )

        plt.xlabel(r'$number$' if scale is None else r'$x$ [$\mu m$]')
        plt.ylabel(r'$I$ [$\%$]')
        plt.grid(color='grey', linestyle=':')

        plt.show()

    #
    return bias


def estimate_fwhm(grid: Grid, handler: Handler | None = None, bias: T = 0, limit: T = 0.5, scale: MicroMeter | None = None, verbose: bool = False, show: bool = False) -> T:
    """Estimate a full width at half maximum (FWHM) of the `grid`.

    Params:
        handler: 
        limit: float - the lowest limit of width

    A grid should be centered!
    """
    if handler is None:
        handler = LinearInterpolationHandler(grid=grid)

    # fwhm
    def _loss(x: T, bias: T, handler: Callable[[float], float]) -> float:
        return (handler(bias)/2 - handler(bias + x))**2

    res = optimize.minimize(
        partial(_loss, bias=bias, handler=handler),
        x0=-limit,
        bounds=[
            (-np.inf, -limit/2),
        ],
    )
    assert res['success'], 'Optimization is not success!'
    lb = res['x'].item()

    res = optimize.minimize(
        partial(_loss, bias=bias, handler=handler),
        x0=+limit,
        bounds=[
            (+limit/2, +np.inf),
        ],
    )
    assert res['success'], 'Optimization is not success!'
    rb = res['x'].item()

    fwhm = rb - lb

    # verbose
    if verbose:
        content = '\n'.join([
            'FWHM: {value:.4f} {units}'.format(
                value=to_scale(fwhm, scale=scale),
                units='' if scale is None else r'[micro]',
            ),
        ])
        print(content)

    # show
    if show:
        fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)

        x, y = grid.x, grid.y
        plt.plot(
            to_scale(x, scale=scale), y,
            color='red', linestyle='none', marker='s', markersize=3,
            alpha=1,
        )

        x = grid.space()
        f_hat = handler(x)
        plt.plot(
            to_scale(x, scale=scale), f_hat,
            color='black', linestyle='-', linewidth=1,
            alpha=1,
        )

        x = np.array([bias + lb, bias + rb])
        y = handler(x)
        plt.plot(
            to_scale(x, scale=scale), y,
            color='red', linestyle='--', linewidth=1,
            alpha=1,
        )

        content = '\n'.join([
            'FWHM: {value:.2f} {units}'.format(
                value=to_scale(fwhm, scale=scale),
                units='' if scale is None else r'[$\mu m$]',
            ),
        ])
        plt.text(
            0.05, 0.95,
            content,
            transform=ax.transAxes,
            ha='left', va='top',
        )

        plt.xlabel(r'$number$' if scale is None else r'$x$ [$\mu m$]')
        plt.ylabel(r'$I$ [$\%$]')
        plt.grid(color='grey', linestyle=':')

        plt.show()

    # 
    return fwhm
