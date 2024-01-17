from abc import ABC
from collections.abc import Sequence
from functools import partial
from typing import Callable, TypeAlias, overload

from scipy import interpolate, optimize
import matplotlib.pyplot as plt

from spectrumlab.alias import Array, Number, MicroMeter
from spectrumlab.core.grid import Grid
from spectrumlab.peak.shape.voight_peak_shape import VoightPeakShape
from spectrumlab.peak.shape.scope import ScopeVariables
from spectrumlab.utils import mse


@overload
def to_scale(__value: Number, scale: MicroMeter) -> MicroMeter: ...
@overload
def to_scale(__value: Number, scale: None) -> Number: ...
def to_scale(__value, scale):
    if scale is None:
        return __value
    
    return scale * __value


class BaseHandler(ABC):

    def __init__(self, grid: Grid, scale: MicroMeter | None = None):
        self._grid = grid
        self._scale = scale

    @property
    def f(self) -> Callable[[Array[Number]], Array[float]]:
        return self._f

    # --------        handlers        --------
    def show(self, bias: Number | None = None):
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
        return self.handler(x)


class LinearInterpolationHandler(BaseHandler):

    def __init__(self, grid: Grid, scale: MicroMeter | None = None, show: bool = False):

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

        # 
        self._f = partial(shape, **scope_variables)

        # show
        if show:
            self.show(bias=scope_variables['position'], scale=scale)


Handler: TypeAlias = LinearInterpolationHandler | VoightPeakShapeHandler
