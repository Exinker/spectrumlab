import warnings
from typing import Callable, TYPE_CHECKING
from typing import overload

import numpy as np
from scipy import interpolate, signal

from spectrumlab.alias import Array, Number
from spectrumlab.emulation.curve import pvoigt, rectangular
from spectrumlab.peak.shape.base_shape import BasePeakShape 
from spectrumlab.peak.shape.voight_peak_shape import VoightPeakShape
from spectrumlab.peak.shape.utils import approx_peak_by_tail

if TYPE_CHECKING:
    from spectrumlab.peak.analyte_peak import AnalytePeak


warnings.filterwarnings('ignore')


class SelfReversedVoigtPeakShapeNaive(BasePeakShape):
    """Self reversed voigt peak's shape type."""
    MAX_EFFECT = 10

    def __init__(self, width: Number, asymmetry: float, ratio: float, rx: Number = 10, dx: Number = .01, re: float = 4, de: float = 1e-1) -> None:
        super().__init__()

        self.width = width
        self.asymmetry = asymmetry
        self.ratio = ratio
        self.rx = rx
        self.dx = dx
        self.re = re
        self.de = de

        self._f = None

    @property
    def f(self) -> Callable[[Array[Number], float], Array[float]]:
        if self._f is None:
            effect = np.linspace(0, self.re, int(self.re/self.de) + 1)

            x = np.linspace(-self.rx, +self.rx, 2*int(self.rx/self.dx) + 1)
            y = np.array([self._apply_effect(x, e) for e in effect])

            self._f = interpolate.interp2d(
                x,
                effect,
                y,
                kind='linear',
                bounds_error=False,
                fill_value=0,
            )

        return self._f

    # --------        approx interface        --------
    def approx_keys(self) -> tuple[str]:
        return (
            'background',
            'position',
            'intensity',
            'effect',
        )

    def approx_initial(self, peak: 'AnalytePeak') -> Array[float]:
        return np.array([
            0,
            peak.position,
            approx_peak_by_tail(
                peak=peak,
                shape=self,
            ),
            0,
        ])

    def approx_bounds(self, peak: 'AnalytePeak', delta: Number = 0) -> tuple[tuple[float, float]]:
        delta += 1e-32  # fix bounds if delta == 0

        return tuple([
            (-1e-10, +1e-10),  # FIXME: нужно разобраться с пределами у фона!
            (peak.position - delta, peak.position + delta),
            (0, np.inf),
            (0, self.MAX_EFFECT),
        ])

    # --------        private        --------
    def _apply_effect(self, x: Array[Number], effect: float) -> Array[float]:
        width = self.width
        asymmetry = self.asymmetry
        ratio = self.ratio

        f = signal.convolve(
            pvoigt(x, x0=0, w=width, a=asymmetry, r=ratio) * 10**(-effect * pvoigt(x, x0=0, w=width, a=asymmetry, r=ratio)),
            rectangular(x, x0=0, w=1),
            mode='same',
        ) * self.dx

        return f

    @overload
    def __call__(self, x: float, position: Number, intensity: float, background: float = 0, effect: float = 0) -> float: ...
    @overload
    def __call__(self, x: Array[Number], position: Number, intensity: float, background: float = 0, effect: float = 0) -> Array[float]: ...
    def __call__(self, x, position, intensity, background=0, effect=0):
        """Interpolate by grip."""
        return background + intensity*self.f(x - position, effect)

    def __repr__(self) -> str:
        cls = self.__class__

        return f'{cls.__name__}(w={self.width:.4f}; a={self.asymmetry:.4f}; r={self.ratio:.4f})'


class SelfReversedVoigtPeakShape(BasePeakShape):
    """Self reversed voigt peak's shape type."""

    def __init__(self, emission_shape: VoightPeakShape, absorption_shape: VoightPeakShape, rx: Number = 10, dx: Number = 1e-2) -> None:
        super().__init__()

        self.emission_shape = emission_shape
        self.absorption_shape = absorption_shape
        self.rx = rx  # границы построения интерполяции
        self.dx = dx  # шаг сетки интерполяции

        self._f = None

    # --------        approx interface        --------
    def approx_keys(self) -> tuple[str]:
        return (
            'background',
            'position',
            'intensity',
            'effect',
        )

    def approx_initial(self, peak: 'AnalytePeak') -> Array[float]:
        return np.array([
            0,
            peak.position,
            approx_peak_by_tail(
                peak=peak,
                shape=self,
            ),
            0,
        ])

    def approx_bounds(self, peak: 'AnalytePeak', delta: Number = 0) -> tuple[tuple[float, float]]:
        delta += 1e-32  # fix bounds if delta == 0

        return tuple([
            (-1e-10, +1e-10),  # FIXME: нужно разобраться с пределами у фона!
            (peak.position - delta, peak.position + delta),
            (0, np.inf),
            (0, self.MAX_EFFECT),
        ])

    # --------        private        --------
    @overload
    def __call__(self, x: float, position: Number, intensity: float, background: float = 0, effect: float = 0) -> float: ...
    @overload
    def __call__(self, x: Array[Number], position: Number, intensity: float, background: float = 0, effect: float = 0) -> Array[float]: ...
    def __call__(self, x, position, intensity, background=0, effect=0):
        """Interpolate by grip."""
        x = np.linspace(-self.rx, +self.rx, 2*int(self.rx/self.dx) + 1)
        y = signal.convolve(
            self.emission_shape(x, 0, 1) * 10**(-effect*self.absorption_shape(x, 0, 1)),
            rectangular(x, x0=0, w=1),
            mode='same',
        ) * self.dx

        f = interpolate.interp1d(
            x,
            y,
            kind='linear',
            bounds_error=False,
            fill_value=0,
        )

        return background + intensity*f(x - position)

    def __repr__(self) -> str:

        return '\n'.join([
            f'    emission: {self.emission_profile}',
            f'    absorption: {self.absorption_profile}',
        ])
