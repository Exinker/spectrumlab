import warnings
from typing import Callable, TYPE_CHECKING
from typing import overload

import numpy as np
from scipy import interpolate, signal

from spectrumlab.curves import pvoigt, rectangular
from spectrumlab.peaks.analyte_peaks.shapes import PeakShape
from spectrumlab.peaks.analyte_peaks.shapes.utils import approx_peak_by_tail
from spectrumlab.types import Array, Number, R

if TYPE_CHECKING:
    from spectrumlab.peaks.analyte_peaks.analyte_peak import AnalytePeak


warnings.filterwarnings('ignore')


class ReversedPeakShapeNaive(PeakShape):

    def __init__(
        self,
        width: Number,
        asymmetry: float,
        ratio: float,
        rx: Number = 10,
        dx: Number = .01,
        re: float = 4,
        de: float = 1e-1,
    ) -> None:
        super().__init__(
            width=width,
            asymmetry=asymmetry,
            ratio=ratio,
            rx=rx,
            dx=dx,
        )

        self.re = re
        self.de = de

        self._f = None

    @property
    def f(self) -> Callable[[Array[Number]], Array[R]]:
        if self._f is None:
            effect = np.linspace(0, self.re, int(self.re/self.de) + 1)

            x = np.linspace(-self.rx, +self.rx, 2*int(self.rx/self.dx) + 1)
            y = np.array([self._apply_effect(x, e) for e in effect])

            self._f = interpolate.RegularGridInterpolator(
                (x, effect),
                y.T,
                method='linear',
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
            approx_peak_by_tail(peak=peak, shape=self),
            0,
        ])

    def approx_bounds(self, peak: 'AnalytePeak', delta: Number = 0) -> tuple[tuple[float, float]]:
        delta += 1e-32  # fix bounds if delta == 0

        return tuple([
            (-1e-10, +1e-10),  # FIXME: нужно разобраться с пределами у фона!
            (peak.position - delta, peak.position + delta),
            (0, np.inf),
            (0, self.re),
        ])

    @overload
    def __call__(
        self,
        x: Number,
        position: Number,
        intensity: float,
        background: float = 0,
        effect: float = 0,
    ) -> R: ...
    @overload
    def __call__(
        self,
        x: Array[Number],
        position: Number,
        intensity: float,
        background: float = 0,
        effect: float = 0,
    ) -> Array[R]: ...
    def __call__(self, x, position, intensity, background=0, effect=0):
        """Interpolate by grip."""

        points = np.column_stack([x - position, np.full_like(x - position, effect)])
        return background + intensity*self.f(points)

    def __repr__(self) -> str:
        cls = self.__class__

        return f'{cls.__name__}(w={self.width:.4f}; a={self.asymmetry:.4f}; r={self.ratio:.4f})'

    def _apply_effect(self, x: Array[Number], effect: float) -> Array[R]:
        width = self.width
        asymmetry = self.asymmetry
        ratio = self.ratio

        f = signal.convolve(
            pvoigt(x, x0=0, w=width, a=asymmetry, r=ratio) * 10**(-effect * pvoigt(x, x0=0, w=width, a=asymmetry, r=ratio)),  # noqa: E501
            rectangular(x, x0=0, w=1),
            mode='same',
        ) * self.dx

        return f


class ReversedPeakShape(PeakShape):

    max_effect = 50

    def __init__(
        self,
        emission_shape: PeakShape,
        absorption_shape: PeakShape | None,
        rx: Number = 10,
        dx: Number = 1e-2,
    ) -> None:

        self.emission_shape = emission_shape
        self.absorption_shape = absorption_shape
        self.rx = rx  # границы построения интерполяции
        self.dx = dx  # шаг сетки интерполяции

        self._f = None

    # --------        approx interface        --------
    def approx_keys(self) -> tuple[str]:

        if self.absorption_shape is None:
            return (
                'background',
                'position',
                'intensity',
                'effect',
                'effect_width',
                'effect_ratio',
            )

        return (
            'background',
            'position',
            'intensity',
            'effect',
        )

    def approx_initial(self, peak: 'AnalytePeak') -> Array[float]:

        if self.absorption_shape is None:
            return np.array([
                0,
                peak.position,
                approx_peak_by_tail(peak=peak, shape=self.emission_shape),
                0,
                2,
                0.1,
            ])

        return np.array([
            0,
            peak.position,
            approx_peak_by_tail(peak=peak, shape=self.emission_shape),
            0,
        ])

    def approx_bounds(self, peak: 'AnalytePeak', delta: Number = 0) -> tuple[tuple[float, float]]:
        delta += 1e-32

        if self.absorption_shape is None:
            return tuple([
                (-1e-10, +1e-10),  # FIXME: нужно разобраться с пределами у фона!
                (peak.position - delta, peak.position + delta),
                (0, np.inf),
                (0, self.max_effect),
                (2, 10),
                (0.1, 1),
            ])

        return tuple([
            (-1e-10, +1e-10),  # FIXME: нужно разобраться с пределами у фона!
            (peak.position - delta, peak.position + delta),
            (0, np.inf),
            (0, self.max_effect),
        ])

    @overload
    def __call__(
        self,
        x: Number,
        position: Number,
        intensity: float,
        background: float = 0,
        effect: float = 0,
        **kwargs,
    ) -> R: ...
    @overload
    def __call__(
        self,
        x: Array[Number],
        position: Number,
        intensity: float,
        background: float = 0,
        effect: float = 0,
        **kwargs,
    ) -> Array[R]: ...
    def __call__(self, x, position, intensity, background=0, effect=0, **kwargs):
        """Interpolate by grip."""
        absorption_shape = self.absorption_shape or PeakShape(
            kwargs['effect_width'],
            0,
            kwargs['effect_ratio'],
            rx=self.emission_shape.rx,
            dx=self.emission_shape.dx,
        )

        _x = np.linspace(-self.rx, +self.rx, 2*int(self.rx/self.dx) + 1)
        _y = signal.convolve(
            self.emission_shape(_x, 0, 1) * 10**(-effect*absorption_shape(_x, 0, 1)),
            rectangular(_x, x0=0, w=1),
            mode='same',
        ) * self.dx
        f = interpolate.interp1d(
            _x,
            _y,
            kind='linear',
            bounds_error=False,
            fill_value=0,
        )

        return background + intensity*f(x - position)

    def __repr__(self) -> str:

        return '\n'.join([
            f'    emission: {self.emission_shape}',
            f'    absorption: {self.absorption_shape}',
        ])
