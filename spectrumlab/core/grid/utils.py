from abc import ABC
from dataclasses import dataclass, field
from functools import partial
from typing import Callable, TypeAlias

import numpy as np
from scipy import interpolate, optimize, signal
import matplotlib.pyplot as plt

from spectrumlab.alias import Array, MicroMeter
from spectrumlab.core.grid import Grid, T
from spectrumlab.emulation.curve import pvoigt, rectangular


@dataclass
class VoigtShape:
    width: T
    asymmetry: float
    ratio: float

    pitch: T
    dx: float = field(default=1e-2)  # шаг построения интерполяции
    rx: float = field(default=10)  # границы построения интерполяции

    def __post_init__(self):
        x = np.linspace(-self.rx, +self.rx, 2*int(self.rx/self.dx) + 1)*self.pitch

        f = lambda x: pvoigt(x, x0=0, w=self.width, a=self.asymmetry, r=self.ratio)
        s = lambda x: rectangular(x, x0=0, w=self.pitch)
        y = signal.convolve(f(x), s(x), mode='same') * self.dx

        self._f = interpolate.interp1d(
            x, y,
            kind='linear',
            bounds_error=False,
            fill_value=0,
        )

    def __call__(self, x: T | Array[T], position: T, intensity: float, background: float = 0) -> Array[float]:
        '''interpolate by grip'''
        f = self._f

        return background + intensity*f(x - position)

    def __repr__(self) -> str:
        return f'{type(self).__name__} (w={self.width:.4f}; a={self.asymmetry:.4f}; r={self.ratio:.4f})'


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
    def show(self, bias: T = 0):
        fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)

        x, y = self.grid.x, self.grid.y
        plt.plot(
            x - bias, y,
            color='red', linestyle='none', marker='s', markersize=3,
            alpha=1,
        )

        x = self.grid.space()
        y_hat = self.f(x)
        plt.plot(
            x - bias, y_hat,
            color='black', linestyle='-', linewidth=1,
            alpha=1,
        )

        x, y = self.grid.x, self.grid.y
        y_hat = self.f(x)
        plt.plot(
            x - bias, y - y_hat,
            color='black', linestyle='none', marker='s', markersize=0.5,
            alpha=1,
        )

        plt.xlabel(self.grid.xlabel)
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


class VoigtShapeHandler(BaseHandler):

    def __init__(self, grid: Grid, pitch: T, show: bool = False):
        super().__init__(grid=grid)

        def _loss(grid: Grid, pitch: T, position: MicroMeter, width: MicroMeter, asymmetry: float, ratio: float, intensity: float) -> float:
            shape = VoigtShape(
                width=width,
                asymmetry=asymmetry,
                ratio=ratio,
                pitch=pitch,
            )

            f = partial(shape, position=position, intensity=intensity)
            return np.sum(
                (grid.y - f(grid.x))**2
            )

        # shape
        x0 = grid.x[np.argmax(grid.y)]
        position, width, asymmetry, ratio, intensity = optimize.minimize(
            lambda x: _loss(grid, pitch, *x),
            x0=[x0, pitch, 0, .1, np.sum(grid.y) / pitch],
            bounds=[(x0-pitch/2, x0+pitch/2), (pitch/2, 100), (-1, 1), (0, 1), (0, np.inf)]
        )['x']

        shape = VoigtShape(
            width=width,
            asymmetry=asymmetry,
            ratio=ratio,
            pitch=pitch,
        )

        # f
        self._f = partial(shape, position=position, intensity=intensity)

        # show
        if show:
            self.show()


Handler: TypeAlias = LinearInterpolationHandler | VoigtShapeHandler


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


def estimate_fwhm(grid: Grid, pitch: T, handler: Handler | None = None, bias: T = 0, verbose: bool = False, show: bool = False) -> T:
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
