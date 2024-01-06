from collections.abc import Sequence
from functools import partial
from typing import overload, Literal, TypeAlias

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate, optimize, signal

from spectrumlab.alias import Array, Number
from spectrumlab.emulation.curve import pvoigt, rectangular
from spectrumlab.utils import mse
from spectrumlab.peak.shape.grid import Grid
from spectrumlab.peak.shape.base_variables import BaseVariables, ScopeVariables, VoightVariables
from spectrumlab.peak.shape.base_shape import BasePeakShape


# --------        voight peak shape        --------
class VoightPeakShapeVariables(BaseVariables):

    def __init__(self, grid: Grid, *args, **kwargs):
        super().__init__([
            VoightVariables(),
            ScopeVariables(grid, *args, **kwargs),
        ])

        #
        self.grid = grid

    @property
    def initial(self) -> tuple[float]:

        result = ()
        for key in self._items.keys():
            item = self._items[key]
            result += item.initial

        return result

    @property
    def bounds(self) -> tuple[tuple[float]]:

        result = ()
        for key in self._items.keys():
            item = self._items[key]
            result += item.bounds

        return result

    @property
    def value(self) -> tuple[tuple[float]]:

        result = ()
        for key in self._items.keys():
            item = self._items[key]
            result += item.value

        return result

    # --------        handlers        --------
    @classmethod
    def parse_params(cls, grid: Grid, params: Sequence[float]) -> tuple[VoightVariables, ScopeVariables]:
        assert len(params) == 6

        shape_variables = VoightVariables(*params[:3])
        scope_variables = ScopeVariables(grid, *params[3:])

        return shape_variables, scope_variables


class VoightPeakShape(BasePeakShape):

    def __init__(self, width: Number, asymmetry: float, ratio: float, rx: Number = 10, dx: Number = 1e-3) -> None:
        """Voight peak's shape. A convolution of apparatus shape and aperture shape (rectangular) of a detector.

        Params:
            width: Number - apparatus shape's width
            asymmetry: float - apparatus shape's asymmetry
            ratio: float - apparatus shape's ratio

            rx: Number = 10 - range of grid
            dx: Number = 0.01 - step of grid
        """
        super().__init__()

        self.width = width
        self.asymmetry = asymmetry
        self.ratio = ratio
        self.rx = rx  # границы построения интерполяции
        self.dx = dx  # шаг сетки интерполяции

        # grid
        x = np.linspace(-self.rx, +self.rx, 2*int(self.rx/self.dx) + 1)

        f = lambda x: pvoigt(x, x0=0, w=self.width, a=self.asymmetry, r=self.ratio)
        s = lambda x: rectangular(x, x0=0, w=1)
        y = signal.convolve(f(x), s(x), mode='same') * (x[-1] - x[0])/len(x)

        self._xvalues = x
        self._yvalues = y

    # --------        fabric        --------
    @classmethod
    def from_grid(cls, grid: Grid, show: bool = False) -> 'VoightPeakShape':

        def loss(grid: Grid, params: Sequence[float]) -> float:

            # variables
            shape_variables, scope_variables = VoightPeakShapeVariables.parse_params(grid=grid, params=params)

            # shape
            shape = VoightPeakShape(**shape_variables)

            # 
            return mse(
                y=grid.yvalues,
                y_hat=shape(x=grid.xvalues, **scope_variables),
            )

        # variables
        variables = VoightPeakShapeVariables(grid=grid)

        res = optimize.minimize(
            partial(loss, grid),
            variables.initial,
            method='SLSQP',
            bounds=variables.bounds,
        )
        assert res['success'], 'Optimization is not succeeded!'

        shape_variables, scope_variables = VoightPeakShapeVariables.parse_params(grid=grid, params=res['x'])

        # shape
        shape = cls(**shape_variables)

        # show
        if show:
            fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)

            x, y = grid.xvalues, grid.yvalues
            plt.plot(
                x, y,
                color='red', linestyle='none', marker='s', markersize=3,
                alpha=1,
            )

            x = np.linspace(min(grid.xvalues), max(grid.xvalues), 1000)
            y_hat = shape(x, **scope_variables)
            plt.plot(
                x, y_hat,
                color='black', linestyle='-', linewidth=1,
                alpha=1,
            )

            x, y = grid.xvalues, grid.yvalues
            y_hat = shape(grid.xvalues, **scope_variables)
            plt.plot(
                x, y - y_hat,
                color='black', linestyle='none', marker='s', markersize=0.5,
                alpha=1,
            )

            content = get_content(shape, sep='\n')
            plt.text(
                0.05, 0.95,
                content,
                transform=ax.transAxes,
                ha='left', va='top',
            )

            plt.xlim([-10, +10])
            plt.xlabel(r'$number$')
            plt.ylabel(r'$I$ [$\%$]')
            plt.grid(color='grey', linestyle=':')

            plt.show()

        # 
        return shape

    # --------        private        --------
    @overload
    def __call__(self, x: Number, position: Number, intensity: float, background: float = 0) -> float: ...
    @overload
    def __call__(self, x: Array[Number], position: Number, intensity: float, background: float = 0) -> Array[float]: ...
    def __call__(self, x, position, intensity, background=0):

        f = interpolate.interp1d(
            self._xvalues,
            self._yvalues,
            kind='linear',
            bounds_error=False,
            fill_value=0,
        )

        return background + intensity*f(x - position)

    def __repr__(self) -> str:
        cls = self.__class__

        return f'{cls.__name__}({get_content(self)})'


class EffectedVoightPeakShape(BasePeakShape):
    """Effected voight peak's shape type."""

    def __init__(self, width: Number, asymmetry: float, ratio: float, rx: Number = 10, dx: Number = .01, de: float = 0.25, re: float = 4) -> None:
        super().__init__()

        self.width = width
        self.asymmetry = asymmetry
        self.ratio = ratio
        self.rx = rx
        self.dx = dx
        self.re = re
        self.de = de

        # grid
        self.xvalues = np.linspace(-self.rx, +self.rx, 2*int(self.rx/self.dx) + 1)
        self.evalues = np.linspace(0, self.re, int(self.re/self.de) + 1)
        self.yvalues = np.array([self._apply_effect(effect=effect) for effect in self.evalues])

    # --------        private        --------
    @overload
    def __call__(self, x: float, position: Number, intensity: float, background: float = 0, effect: float = 0) -> float: ...
    @overload
    def __call__(self, x: Array[float], position: Number, intensity: float, background: float = 0, effect: float = 0) -> Array[float]: ...
    def __call__(self, x, position, intensity, background=0, effect=0):
        """Interpolate by grip."""

        f = interpolate.interp2d(
            self.xvalues,
            self.evalues,
            self.yvalues,
            kind='linear',
            bounds_error=False,
            fill_value=0,
        )

        return background + intensity*f(x - position, effect)

    def __repr__(self) -> str:
        cls = self.__class__

        return f'{cls.__name__}(w={self.width:.4f}; a={self.asymmetry:.4f}; r={self.ratio:.4f})'

    def _apply_effect(self, effect: float) -> Array[float]:
        width = self.width
        asymmetry = self.asymmetry
        ratio = self.ratio

        x = np.linspace(-self.rx, +self.rx, 2*int(self.rx/self.dx) + 1)

        f = lambda x: pvoigt(x, x0=0, w=width, a=asymmetry, r=ratio)
        g = lambda x: pvoigt(x, x0=0, w=width, a=asymmetry, r=ratio)
        h = lambda x: rectangular(x, x0=0, w=1)

        return signal.convolve(f(x) * 10**(-effect * g(x)), h(x), mode='same') * (x[-1] - x[0])/len(x)


PeakShape: TypeAlias = VoightPeakShape | EffectedVoightPeakShape


def get_content(p: PeakShape, sep: Literal[r'\n', '; '] = '; ', is_signed: bool = True) -> str:
    sign = {+1: '+' }.get(np.sign(p.asymmetry), '') if is_signed else ''

    return sep.join([
        f'w={p.width:.4f}',
        f'a={sign}{p.asymmetry:.4f}',
        f'r={p.ratio:.4f}',
    ])


# --------        handlers        --------
def approx_grid(grid: Grid, shape: VoightPeakShape, show: bool = False) -> tuple[ScopeVariables, float]:
    """Approximate grid by VoightPeakShape."""

    def _fitness(params: Sequence[float], grid: Grid, shape: VoightPeakShape) -> float:
        scope_variables = ScopeVariables(grid, *params)

        y = grid.yvalues
        y_hat = shape(x=grid.xvalues, **scope_variables)

        return mse(y, y_hat)

    # variables
    variables = ScopeVariables(grid=grid)

    res = optimize.minimize(
        partial(_fitness, grid=grid, shape=shape),
        variables.initial,
        method='SLSQP',
        bounds=variables.bounds,
    )
    assert res['success'], 'Optimization is not succeeded!'

    scope_variables = ScopeVariables(grid, *res['x'])

    #
    y = grid.yvalues
    y_hat = shape(x=grid.xvalues, **scope_variables)

    error = mse(y, y_hat) / scope_variables['intensity']

    # show
    if show:
        plt.subplots(figsize=(6, 4), tight_layout=True)

        plt.title(f'error: {error:.5f}')

        x, y = grid.xvalues, grid.yvalues
        plt.plot(
            x, y,
            color='red', linestyle='none', marker='s', markersize=3,
            alpha=1,
        )

        x = np.linspace(min(grid.xvalues), max(grid.xvalues), 1000)
        y_hat = shape(x, **scope_variables)
        plt.plot(
            x, y_hat,
            color='black', linestyle='-', linewidth=1,
            alpha=1,
        )

        x, y = grid.xvalues, grid.yvalues
        y_hat = shape(x, **scope_variables)
        plt.plot(
            x, y - y_hat,
            color='black', linestyle='none', marker='s', markersize=0.5,
            alpha=1,
        )

        plt.xlim([-10, +10])
        plt.xlabel(r'$number$')
        plt.ylabel(r'$I$ [$\%$]')
        plt.grid(color='grey', linestyle=':')

        plt.show()

    #
    return scope_variables, error
