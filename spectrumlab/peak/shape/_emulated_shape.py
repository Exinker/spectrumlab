
from abc import ABC
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import partial
from typing import overload, TypeAlias, Type

from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate, optimize, signal

from spectrumlab.alias import Array, Number
from spectrumlab.emulation.emulation import Emulation
from spectrumlab.emulation.curve import pvoigt, rectangular
from spectrumlab.utils import mse
from spectrumlab.peak.shape.grid import Grid
from spectrumlab.peak.shape.peak_shape import VoightPeakShape


class EmulatedPeakShape(VoightPeakShape):

    def __init__(self, emulation: Emulation, position: Number, concentrations: tuple[float]) -> None:
        super().__init__()

        self.emulation = emulation
        self.position = position
        self.concentrations = concentrations

        # grid
        x = np.arange(-self.rx, self.rx+self.dx, self.dx)
        y = np.arange(0, self.ry+self.dy, self.dy)

        f = lambda x: pvoigt(x, x0=0, w=self.width, a=self.asymmetry, r=self.ratio)
        h = lambda x: rectangular(x, x0=0, w=1)
        y = signal.convolve(f(x), h(x), mode='same') * (x[-1] - x[0])/len(x)

        self._xvalues = x
        self._yvalues = y

        #
        frames = []
        for i, concentration in enumerate(tqdm(concentrations)):
            emulation = emulation.setup(position=position, concentration=concentration)
            spectrum = emulation.run(is_noised=False, is_clipped=False)

            frames.append(spectrum.intensity)

    @overload
    def __call__(self, x: float, position: Number, intensity: float, background: float = 0, effect_value: float = 0) -> float: ...
    @overload
    def __call__(self, x: Array[float], position: Number, intensity: float, background: float = 0, effect_value: float = 0) -> Array[float]: ...
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
