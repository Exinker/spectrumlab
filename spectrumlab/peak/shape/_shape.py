from abc import ABC
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import partial
from typing import overload, TypeAlias

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate, optimize, signal

from spectrumlab.alias import Array, Number
from spectrumlab.emulation.curve import pvoigt, rectangular
from spectrumlab.utils import mse
from spectrumlab.peak.shape.grid import Grid


@dataclass
class Variable:
    name: str
    initial: float
    bounds: tuple[float, float]
    value: float | None = None


class Variables(ABC):
    data: Mapping[str, Variable]

    def __getitem__(self, key: str):
        value = self.data[key].value

        return value

    def keys(self) -> tuple[str]:
        return tuple(dat.name for dat in self.data.values())

    @property
    def initial(self) -> tuple:
        return tuple(dat.initial for dat in self.data.values())

    @property
    def bounds(self) -> tuple[tuple[float, float]]:
        return tuple(dat.bounds for dat in self.data.values())

    @property
    def value(self) -> tuple[float]:
        return tuple(dat.value for dat in self.data.values())


class VoightVariables(Variables):
    _keys = ('width', 'asymmetry', 'ratio')  # TODO: refactor!

    def __init__(self):
        self.data = {
            'width': Variable('width', 2., (0.1, 20)),
            'asymmetry': Variable('asymmetry', .0, (-.5, +.5)),
            'ratio': Variable('ratio', .1, (0, 1)),
        }


class ScopeVariables(Variables):
    _keys = ('position', 'intensity', 'background')  # TODO: refactor!

    def __init__(self, grid: Grid, position: float | None = None, intensity: float | None = None, background: float | None = None):

        p0 = grid.x[np.argmax(grid.y)] if position is None else position
        i0 = max(grid.y) if intensity is None else intensity
        b0 = min(grid.y) if position is None else background

        self.data = {
            'position': Variable(
                'position',
                p0,
                (p0 - 2, p0 + 2) if position is None else (p0 - 1e-10, p0 + 1e-10),
            ),
            'intensity': Variable(
                'intensity',
                i0,
                (0, +np.inf) if intensity is None else (i0 - 1e-10, i0 + 1e-10),
            ),
            'background': Variable(
                'background',
                b0,
                (min(grid.y), max(grid.y)) if background is None else (b0 - 1e-10, b0 + 1e-10),
            ),
        }


class ApproxVariables:
    
    def __init__(self, grid: Grid, *args, **kwargs):
        self.data = {
            'shape': VoightVariables(),
            'scope': ScopeVariables(grid, *args, **kwargs),
        }
        self._values = None

    def keys(self) -> tuple[str]:
        res = tuple()
        for dat in self.data.values():
            res += dat.keys()

        return res

    @property
    def initial(self) -> tuple:
        res = tuple()
        for dat in self.data.values():
            res += dat.initial

        return res

    @property
    def bounds(self) -> tuple:
        res = tuple()
        for dat in self.data.values():
            res += dat.bounds

        return res

    def set_values(self, values) -> None:
        for dat, value in zip(self.data.values(), values):
            dat.value = value

    @property
    def values(self) -> tuple:
        return sum([dat.value for dat in self.data.values()])

    @staticmethod
    def parse_values(values: Sequence[float]) -> tuple[Mapping[str, float], Mapping[str, float]]:

        shape_variables = {key: value for key, value in zip(VoightVariables._keys, values[:3])}  # TODO: refactor!
        scope_variables = {key: value for key, value in zip(ScopeVariables._keys, values[3:])}  # TODO: refactor!

        return shape_variables, scope_variables


@dataclass
class VoightPeakShape:
    width: float
    asymmetry: float
    ratio: float

    rx: Number = field(default=10)  # границы построения интерполяции
    dx: Number = field(default=.01)  # шаг сетки интерполяции

    dx: float = field(default=0.01)
    rx: float = field(default=20)

    _x_grid: Array[float] = field(init=False, repr=False)
    _y_grid: Array[float] = field(init=False, repr=False)

    def __post_init__(self):
        x = np.linspace(-self.rx, +self.rx, 2*int(self.rx/self.dx) + 1)

        f = lambda x: pvoigt(x, x0=0, w=self.width, a=self.asymmetry, r=self.ratio)
        h = lambda x: rectangular(x, x0=0, w=1)
        y = signal.convolve(f(x), h(x), mode='same') * (x[-1] - x[0])/len(x)

        self._x_grid = x
        self._y_grid = y

    @overload
    def __call__(self, x: float, position: float, intensity: float, background: float = 0) -> float: ...
    @overload
    def __call__(self, x: Array[float], position: float, intensity: float, background: float = 0) -> Array[float]: ...
    def __call__(self, x, position, intensity, background=0):

        f = interpolate.interp1d(
            self._x_grid,
            self._y_grid,
            kind='linear',
            bounds_error=False,
            fill_value=0,
        )

        return background + intensity*f(x - position)

    def __repr__(self) -> str:
        cls = self.__class__

        return f'{cls.__name__}(w={self.width:.4f}; a={self.asymmetry:.4f}; r={self.ratio:.4f})'

    # --------        handlers        --------
    @classmethod
    def from_grid(cls, grid: Grid, position: float | None = None, intensity: float | None = None, background: float | None = None, full: bool = False, show: bool = False) -> 'VoightPeakShape':

        #
        variables = ApproxVariables(grid, position=position, intensity=intensity, background=background)

        result = optimize.minimize(
            partial(calculate_approx_fitness, grid),
            variables.initial,
            method='SLSQP',
            bounds=variables.bounds,
        )

        shape_variables, scope_variables = ApproxVariables.parse_values(result.x)

        #
        shape = VoightPeakShape(**shape_variables)

        #
        if show:
            plt.subplots(figsize=(6, 4), tight_layout=True)

            x, y = grid.xvalues, grid.yvalues
            plt.plot(
                x, y,
                color='red', linestyle='none', marker='s', markersize=3,
                alpha=1,
            )

            x = np.linspace(min(grid.x), max(grid.x), 1000)
            y_hat = shape(x, **scope_variables)
            plt.plot(
                x, y_hat,
                color='black', linestyle=':',
            )

            x, y = grid.xvalues, grid.yvalues
            y_hat = shape(grid.x, **scope_variables)
            plt.plot(
                x, y - y_hat,
                color='black', linestyle='none', marker='s', markersize=0.5,
            )

            plt.xlabel(r'$number$')
            plt.ylabel(r'$I$ [$\%$]')
            plt.grid(color='grey', linestyle=':')

            plt.show()

        #
        if full:
            return shape, *scope_variables.values()

        return shape


@dataclass
class SelfReversedVoightPeakShape:
    emission_shape: VoightPeakShape
    absorption_shape: VoightPeakShape | None = field(default=None)

    lim: float = field(default=100)  # границы построения интерполяции

    def __call__(self, x: float | Array, position: float, intensity: float, absorption: float = 0, background: float = 0, width_absorption: float = 2, ratio_absorption: float = 0) -> Array:
        """Interpolate by grip."""

        shape = self.emission_shape
        F_emission = interpolate.interp1d(
            shape.x,
            shape.f,
            kind='linear',
            bounds_error=False,
            fill_value=0,
        )

        shape = VoightPeakShape(
            width=width_absorption,
            asymmetry=0,
            ratio=ratio_absorption,
            lim=200,
        )
        F_absorption = interpolate.interp1d(
            shape.x,
            shape.f,
            kind='linear',
            bounds_error=False,
            fill_value=0,
        )

        f = background + intensity*(F_emission(x - position)*10**(-absorption*F_absorption(x - position)))

        return f

    def __repr__(self) -> str:
        cls = self.__class__

        content = '\n'.join([
            f'\temission: {self.emission_shape}',
            f'\tabsorption: {self.absorption_shape}',
        ])
        return f'{cls.__name__}({content})'


PeakShape: TypeAlias = VoightPeakShape | SelfReversedVoightPeakShape


# --------        PeakShape Approximation        --------
def calculate_approx_fitness(grid: Grid, params: Array) -> float:
    x, y = grid.xvalues, grid.yvalues
    shape_variables, scope_variables = ApproxVariables.parse_values(params)

    y_hat = VoightPeakShape(**shape_variables)(x, **scope_variables)

    return mse(y, y_hat)


def _calculate_fitness(grid: Grid, shape: PeakShape, params: Array) -> float:
    x, y = grid.xvalues, grid.yvalues
    y_hat = shape(x, *params)

    return mse(y, y_hat)


def calculate_approx_scope(grid: Grid, shape: VoightPeakShape, show: bool = False) -> tuple[float, float, float, float]:
    """"""

    #
    variables = ScopeVariables(grid=grid)

    result = optimize.minimize(
        partial(_calculate_fitness, grid, shape),
        variables.initial,
        method='SLSQP',
        bounds=variables.bounds,
    )
    scope_variables = result.x

    #
    x, y = grid.xvalues, grid.yvalues
    y_hat = shape(x, *scope_variables)
    # error = mse(y, y_hat)

    intensity = scope_variables[1]
    error = max(abs((y - y_hat) / intensity))

    if show:
        print(abs((y - y_hat) / intensity))

    #
    if show:
        plt.subplots(figsize=(6, 4), tight_layout=True)

        plt.title(f'error: {error:.5f}')

        x, y = grid.xvalues, grid.yvalues
        plt.plot(
            x, y,
            color='red', linestyle='none', marker='s', markersize=3,
            alpha=1,
        )

        x = np.linspace(min(grid.x), max(grid.x), 1000)
        y_hat = shape(x, *scope_variables)
        plt.plot(
            x, y_hat,
            color='black', linestyle=':',
        )

        x, y = grid.xvalues, grid.yvalues
        y_hat = shape(grid.x, *scope_variables)
        plt.plot(
            x, y - y_hat,
            color='black', linestyle='none', marker='s', markersize=0.5,
        )

        plt.xlabel(r'$number$')
        plt.ylabel(r'$I$ [$\%$]')
        plt.grid(color='grey', linestyle=':')

        plt.show()

    #
    return *scope_variables, error


# --------        utils        --------
def find_fwhm(shape: VoightPeakShape) -> float:
    """Find full width at half maximum (FWHM) for given shape."""
    f = partial(shape, position=0, intensity=1, background=0)

    fwhm = 0
    for x0, bounds in zip([-1, 1], [[(-10*shape.width, 0)], [(0, +10*shape.width)]]):
        res = optimize.minimize(
            lambda x: mse(f(0)/2, f(x)),
            x0=x0,
            bounds=bounds,
            method='Nelder-Mead',
        )
        assert res['success'], 'Optimization is not succeeded!'

        fwhm += np.abs(res['x']).item()

    return fwhm
