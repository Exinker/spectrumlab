from dataclasses import dataclass, field
from typing import TypeAlias

import numpy as np
from scipy import interpolate, signal

from spectrumlab.grid import T
from spectrumlab.emulation.curve import pvoigt, rectangular
from spectrumlab.typing import Array


@dataclass
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

    def __call__(self, x: T | Array[T], position: T, intensity: float, background: float = 0) -> Array[float]:
        """interpolate by grip"""
        f = self._f

        return background + intensity*f(x - position)

    def __repr__(self) -> str:
        return f'{type(self).__name__} (w={self.width:.4f}; a={self.asymmetry:.4f}; r={self.ratio:.4f})'


GridShape: TypeAlias = VoigtGridShape
