from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy import interpolate, signal

from spectrumlab.curves import pvoigt, rectangular
from spectrumlab.types import Array, T


@dataclass(slots=True)
class VoigtGridShape:

    width: T
    asymmetry: float
    ratio: float

    pitch: T
    dx: float = field(default=1e-2)  # шаг построения интерполяции
    rx: float = field(default=10)  # границы построения интерполяции

    def __post_init__(self):
        x = np.linspace(-self.rx, +self.rx, 2*int(self.rx/self.dx) + 1)*self.pitch
        y = signal.convolve(
            pvoigt(x, x0=0, w=self.width, a=self.asymmetry, r=self.ratio),
            rectangular(x, x0=0, w=self.pitch),
            mode='same',
        ) * self.dx

        self._f = interpolate.interp1d(
            x, y,
            kind='linear',
            bounds_error=False,
            fill_value=0,
        )

    def _get_content(self, sep: Literal[r'\n', ', '] = ', ', is_signed: bool = True) -> str:
        sign = {+1: '+'}.get(np.sign(self.asymmetry), '') if is_signed else ''

        return sep.join([
            f'width={self.width:.4f}',
            f'asymmetry={sign}{self.asymmetry:.4f}',
            f'ratio={self.ratio:.4f}',
        ])

    def __call__(self, x: T | Array[T], position: T, intensity: float, background: float = 0) -> Array[float]:
        """interpolate by grip"""
        f = self._f

        return background + intensity*f(x - position)

    def __repr__(self) -> str:
        return f'{type(self).__name__}({self._get_content()})'
