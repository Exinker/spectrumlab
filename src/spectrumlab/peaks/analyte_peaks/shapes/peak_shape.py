from collections.abc import Mapping
from typing import Any, Callable, Literal, TYPE_CHECKING, overload

import numpy as np
from scipy import interpolate, signal

from spectrumlab.curves import pvoigt, rectangular
from spectrumlab.peaks.analyte_peaks.shapes.utils import approx_peak_by_tail
from spectrumlab.types import Array, Number, R

if TYPE_CHECKING:
    from spectrumlab.peaks.analyte_peaks.analyte_peak import AnalytePeak


class PeakShape:

    def __init__(
        self,
        width: Number,
        asymmetry: float,
        ratio: float,
        rx: Number = 10,
        dx: Number = 1e-2,
    ) -> None:
        """Voigt peak's shape. A convolution of apparatus shape and aperture shape (rectangular) of a detector.

        Params:
            width: Number - apparatus shape's width
            asymmetry: float - apparatus shape's asymmetry
            ratio: float - apparatus shape's ratio

            rx: Number = 10 - range of convolution grid
            dx: Number = 0.01 - step of convolution grid
        """
        super().__init__()

        self.width = width
        self.asymmetry = asymmetry
        self.ratio = ratio
        self.rx = rx  # границы построения интерполяции
        self.dx = dx  # шаг сетки интерполяции

        # grid
        x = np.linspace(-self.rx, +self.rx, 2*int(self.rx/self.dx) + 1)
        y = signal.convolve(
            pvoigt(x, x0=0, w=self.width, a=self.asymmetry, r=self.ratio),
            rectangular(x, x0=0, w=1),
            mode='same',
        ) * self.dx

        self._f = interpolate.interp1d(
            x, y,
            kind='linear',
            bounds_error=False,
            fill_value=0,
        )

    @property
    def f(self) -> Callable[[Array[Number]], Array[R]]:
        return self._f

    def get_content(
        self,
        sep: Literal[r'\n', '; '] = '; ',
        is_signed: bool = True,
        fields: Mapping[str, Any] | None = None,
    ) -> str:
        fields = fields or {}
        sign = {+1: '+'}.get(np.sign(self.asymmetry), '') if is_signed else ''

        return sep.join([
            f'{key}={value}'
            for key, value in fields.items()
        ] + [
            f'width={self.width:.4f}',
            f'asymmetry={sign}{self.asymmetry:.4f}',
            f'ratio={self.ratio:.4f}',
        ])

    # --------        approx interface        --------
    def approx_keys(self) -> tuple[str]:
        return (
            'background',
            'position',
            'intensity',
        )

    def approx_initial(self, peak: 'AnalytePeak') -> Array[float]:
        return np.array([
            0,
            peak.position,
            approx_peak_by_tail(peak=peak, shape=self),
        ])

    def approx_bounds(self, peak: 'AnalytePeak', delta: Number = 0) -> tuple[tuple[float, float]]:
        delta += 1e-32  # fix bounds (if `delta` is equal zero)

        return tuple([
            (-1e-10, +1e-10),  # FIXME: нужно разобраться с пределами у фона!
            (peak.position - delta, peak.position + delta),
            (0, np.inf),
        ])

    def parse_params(self, params: Array[float]) -> Mapping[str, float]:
        assert len(self.approx_keys()) == len(params)

        return {
            key: param
            for key, param in zip(self.approx_keys(), params)
        }

    @overload
    def __call__(self, x: Number, position: Number, intensity: float, background: float = 0) -> R: ...
    @overload
    def __call__(self, x: Array[Number], position: Number, intensity: float, background: float = 0) -> Array[R]: ...
    def __call__(self, x, position, intensity, background=0):
        return background + intensity*self.f(x - position)

    def __repr__(self) -> str:
        cls = self.__class__

        return f'{cls.__name__}({self.get_content()})'
