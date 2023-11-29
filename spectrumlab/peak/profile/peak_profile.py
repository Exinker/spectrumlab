
from collections.abc import Sequence
from functools import partial
from typing import overload, TypeAlias

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate, optimize, signal

from spectrumlab.alias import Array, Number
from spectrumlab.emulation.curve import pvoigt, rectangular
from spectrumlab.utils import mse
from spectrumlab.peak.profile.grid import Grid
from spectrumlab.peak.profile.base_variables import BaseVariables, ScopeVariables, VoightVariables
from spectrumlab.peak.profile.base_profile import BaseProfile


class VoightPeakProfileVariables(BaseVariables):

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

    @classmethod
    def parse_params(cls, grid: Grid, params: Sequence[float]) -> tuple[VoightVariables, ScopeVariables]:
        assert len(params) == 6

        profile_variables = VoightVariables(*params[:3])
        scope_variables = ScopeVariables(grid, *params[3:])

        return profile_variables, scope_variables


class VoightPeakProfile(BaseProfile):
    """Voight peak's profile type."""

    def __init__(self, width: Number, asymmetry: float, ratio: float, rx: Number = 10, dx: Number = .01) -> None:
        super().__init__()

        self.width = width
        self.asymmetry = asymmetry
        self.ratio = ratio
        self.rx = rx  # границы построения интерполяции
        self.dx = dx  # шаг сетки интерполяции

        # grid
        x = np.arange(-self.rx, self.rx+self.dx, self.dx)

        f = lambda x: pvoigt(x, x0=0, w=self.width, a=self.asymmetry, r=self.ratio)
        h = lambda x: rectangular(x, x0=0, w=1)
        y = signal.convolve(f(x), h(x), mode='same') * (x[-1] - x[0])/len(x)

        self._xvalues = x
        self._yvalues = y

    @overload
    def __call__(self, x: float, position: Number, intensity: float, background: float = 0) -> float: ...
    @overload
    def __call__(self, x: Array[float], position: Number, intensity: float, background: float = 0) -> Array[float]: ...
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

        return f'{cls.__name__}(w={self.width:.4f}; a={self.asymmetry:.4f}; r={self.ratio:.4f})'

    # --------        fabric        --------
    @classmethod
    def from_grid(cls, grid: Grid, show: bool = False) -> 'VoightPeakProfile':

        def _fitness(grid: Grid, params: Sequence[float]) -> float:
            """Calculate error (fitness) of approximation."""

            # variables
            profile_variables, scope_variables = VoightPeakProfileVariables.parse_params(grid=grid, params=params)

            # profile
            profile = VoightPeakProfile(**profile_variables)

            # 
            return mse(
                y=grid.yvalues,
                y_hat=profile(x=grid.xvalues, **scope_variables),
            )

        # variables
        variables = VoightPeakProfileVariables(grid=grid)

        result = optimize.minimize(
            partial(_fitness, grid),
            variables.initial,
            method='SLSQP',
            bounds=variables.bounds,
        )

        profile_variables, scope_variables = VoightPeakProfileVariables.parse_params(grid=grid, params=result.x)

        # profile
        profile = cls(**profile_variables)

        # show
        if show:
            plt.subplots(figsize=(6, 4), tight_layout=True)

            x, y = grid
            plt.plot(
                x, y,
                color='red', linestyle='none', marker='s', markersize=3,
                alpha=1,
            )

            x = np.linspace(min(grid.x), max(grid.x), 1000)
            y_hat = profile(x, **scope_variables)
            plt.plot(
                x, y_hat,
                color='black', linestyle=':',
            )

            x, y = grid
            y_hat = profile(grid.x, **scope_variables)
            plt.plot(
                x, y - y_hat,
                color='black', linestyle='none', marker='s', markersize=0.5,
            )

            plt.xlabel(r'$number$')
            plt.ylabel(r'$I, \%$')
            plt.grid(color='grey', linestyle=':')

            plt.show()

        # 
        return profile


class EffectedVoightPeakProfile(BaseProfile):
    """Effected voight peak's profile type."""

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
        self.xvalues = np.arange(-self.rx, self.rx+self.dx, self.dx)
        self.evalues = np.arange(0, self.re+self.de, self.de)
        self.yvalues = np.array([self._apply_effect(effect=effect) for effect in self.evalues])

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

        x = np.arange(-self.rx, self.rx+self.dx, self.dx)

        f = lambda x: pvoigt(x, x0=0, w=width, a=asymmetry, r=ratio)
        g = lambda x: pvoigt(x, x0=0, w=width, a=asymmetry, r=ratio)
        h = lambda x: rectangular(x, x0=0, w=1)

        return signal.convolve(f(x) * 10**(-effect * g(x)), h(x), mode='same') * (x[-1] - x[0])/len(x)


PeakProfile: TypeAlias = VoightPeakProfile | EffectedVoightPeakProfile


# --------        handlers        --------
def approx_grid(grid: Grid, profile: VoightPeakProfile, show: bool = False) -> tuple[ScopeVariables, float]:
    """Approximate grid by VoightPeakProfile."""

    def _fitness(params: Sequence[float], grid: Grid, profile: VoightPeakProfile) -> float:
        scope_variables = ScopeVariables(grid, *params)

        y = grid.yvalues
        y_hat = profile(x=grid.xvalues, **scope_variables)

        return mse(y, y_hat)

    # variables
    variables = ScopeVariables(grid=grid)

    result = optimize.minimize(
        partial(_fitness, grid=grid, profile=profile),
        variables.initial,
        method='SLSQP',
        bounds=variables.bounds,
    )

    scope_variables = ScopeVariables(grid, *result.x)

    #
    y = grid.yvalues
    y_hat = profile(x=grid.xvalues, **scope_variables)

    error = mse(y, y_hat)

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
        y_hat = profile(x, **scope_variables)
        plt.plot(
            x, y_hat,
            color='black', linestyle=':',
        )

        x, y = grid.xvalues, grid.yvalues
        y_hat = profile(x, **scope_variables)
        plt.plot(
            x, y - y_hat,
            color='black', linestyle='none', marker='s', markersize=0.5,
        )

        plt.xlabel(r'$number$')
        plt.ylabel(r'$I, \%$')
        plt.grid(color='grey', linestyle=':')

        plt.show()

    #
    return scope_variables, error  # variables.parse_params(grid=grid, params=)
