from dataclasses import dataclass, field
from typing import overload

import numpy as np
from scipy import interpolate, signal

from spectrumlab.curves import pvoigt, rectangular  # noqa: I100
from spectrumlab.peaks.analyte_peaks.shapes import PeakShape
from spectrumlab.types import Array, Number


@dataclass
class SelfReversedEffect:

    width: Number
    asymmetry: float
    ratio: float

    def __call__(self, value: float):
        width = self.width
        asymmetry = self.asymmetry
        ratio = self.ratio

        x = np.linspace(-self.shape.rx, self.shape.rx, 2*int(self.shape.rx/self.shape.dx) + 1)

        f = lambda x: pvoigt(x, x0=0, w=width, a=asymmetry, r=ratio)  # noqa: E731
        g = lambda x: pvoigt(x, x0=0, w=width, a=asymmetry, r=ratio)  # noqa: E731
        h = lambda x: rectangular(x, x0=0, w=1)  # noqa: E731

        return signal.convolve(f(x) * 10**(-value * g(x)), h(x), mode='same') * (x[-1] - x[0])/len(x)


####
@dataclass
class Effect:

    frames: tuple

    def __init__(self):
        pass


@dataclass
class EffectedVoigtPeakShape(PeakShape):

    width: Number
    asymmetry: float
    ratio: float

    dx: float = field(default=0.01)
    rx: float = field(default=20)
    dy: float = field(default=0.25)
    ry: float = field(default=4)

    _x_grid: Array[float] = field(init=False, repr=False)
    _y_grid: Array[float] = field(init=False, repr=False)
    _z_grid: Array[float] = field(init=False, repr=False)

    def __post_init__(self):
        self._x_grid = np.linspace(-self.rx, +self.rx, 2*int(self.rx/self.dx) + 1)
        self._y_grid = np.linspace(0, self.ry, int(self.ry/self.dy) + 1)
        self._z_grid = np.array([self.effect(value=value) for value in self._y_grid])

    @overload
    def __call__(
        self,
        x: float,
        position: Number,
        intensity: float,
        background: float = 0,
        effect_value: float = 0,
    ) -> float: ...
    @overload
    def __call__(
        self,
        x: Array[float],
        position: Number,
        intensity: float,
        background: float = 0,
        effect_value: float = 0,
    ) -> Array[float]: ...
    def __call__(self, x, position, intensity, background=0, effect_value=0):
        """Interpolate by grip."""

        f = interpolate.interp2d(
            self._x_grid,
            self._y_grid,
            self._z_grid,
            kind='linear',
            bounds_error=False,
            fill_value=0,
        )

        return background + intensity*f(x - position, effect_value)

    def __repr__(self) -> str:
        cls = self.__class__

        return f'{cls.__name__}(w={self.width:.4f}; a={self.asymmetry:.4f}; r={self.ratio:.4f})'

    # @classmethod
    # def from_emulation(
    #     cls,
    #     emulation: Emulation,
    #     position: Number,
    #     concentrations: tuple[float],
    # ) -> 'EffectedVoigtPeakShape':

    #     frames = []
    #     for concentration in tqdm(concentrations):
    #         emulation = emulation.setup(position=position, concentration=concentration)
    #         spectrum = emulation.run(is_noised=False, is_clipped=False)

    #         frames.append(spectrum.intensity)


# if __name__ == '__main__':
#     dy = 0.1
#     effect_values = np.arange(0, 1+1, dy)

#     width = 2.0
#     asymmetry = 0
#     ratio = .2

#     #
#     effect = SelfReversedEffect(
#         width=width,
#         asymmetry=asymmetry,
#         ratio=ratio,
#     )

#     shape = EffectedVoigtPeakShape(
#         width=width,
#         asymmetry=asymmetry,
#         ratio=ratio,
#         effect=effect,
#         dy=dy,
#     )
#     shape_hat = EffectedVoigtPeakShape(
#         width=width,
#         asymmetry=asymmetry,
#         ratio=ratio,
#         effect=effect,
#         dy=.25,
#     )

#     fig, (ax_left, ax_right) = plt.subplots(nrows=1, ncols=2, figsize=(12, 4), tight_layout=True)

#     #
#     plt.sca(ax_left)

#     errors = []
#     for effect_value in effect_values:
#         x = np.linspace(-10, 10, 200)
#         y = shape(x, position=0, intensity=1, effect_value=effect_value)
#         plt.plot(
#             x, y,
#             color='black', linestyle='-', linewidth=1,
#         )

#         x = np.linspace(-10, 10, 200)
#         y_hat = shape_hat(x, position=0, intensity=1, effect_value=effect_value)
#         plt.plot(
#             x, y_hat,
#             color='red', linestyle=':', linewidth=1,
#         )

#         errors.append(max(np.abs(100*(y - y_hat) / y)))

#     plt.xlabel(r'x')
#     plt.ylabel(r'y')
#     plt.grid(
#         color='grey', linestyle=':',
#     )

#     #
#     plt.sca(ax_right)

#     x, y = effect_values, errors
#     plt.plot(
#         x, y,
#         color='black', linestyle='-', linewidth=1,
#     )

#     plt.xlabel(r'effect value')
#     plt.ylabel(r'error, %')
#     plt.grid(
#         color='grey', linestyle=':',
#     )

#     #
#     plt.show()
