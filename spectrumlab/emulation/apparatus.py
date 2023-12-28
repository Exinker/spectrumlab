from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

from spectrumlab.alias import Array, Micro
from spectrumlab.emulation.curve import rectangular, pvoigt


# --------        shape        --------
@dataclass
class RectangularApparatusShape:
    """Rectangular device's apparatus shape."""
    width: Micro

    def __call__(self, x: Micro | Array[Micro], x0: Micro) -> Array[float]:
        f = rectangular(x, x0=x0, w=self.width)

        return f


@dataclass
class TriangularApparatusShape:
    """Triangular device's apparatus shape."""
    width: Micro

    def __call__(self, x: Micro | Array[Micro], x0: Micro) -> Array[float]:
        raise NotImplementedError


@dataclass 
class VoigtApparatusShape:
    """Voigt device's apparatus shape.

    A simple asymmetric line shape profile for fitting infrared absorption spectra.
    Aaron L. Stancik, Eric B. Brauns
    https://www.sciencedirect.com/science/article/abs/pii/S0924203108000453
    """
    width: Micro
    asymmetry: float
    ratio: float

    def __call__(self, x: Micro | Array[Micro], x0: Micro) -> Array[float]:
        f = pvoigt(x, x0=x0, w=self.width, a=self.asymmetry, r=self.ratio)

        return f


ApparatusShape = RectangularApparatusShape | TriangularApparatusShape | VoigtApparatusShape


# --------        apparatus        --------
@dataclass
class Apparatus:
    """
    Interface for any apparatus's shape.

    Author: Vaschenko Pavel
     Email: vaschenko@vmk.ru
      Date: 2014.03.24
    """
    shape: ApparatusShape

    def __call__(self, x: Micro | Array[Micro], x0: Micro) -> Array[float]:
        return self.shape(x, x0=x0)

    # --------        handlers        --------
    def show(self, rx: Micro = 100, dx: Micro = .01) -> None:
        
        #
        fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)

        x = np.arange(-rx, +rx+dx, dx)
        y = self(x, x0=0)
        plt.plot(
            x, y,
            color='k',
            label=r'$F(x)$',
        )

        plt.xlabel(r'$x, \mu$')
        plt.ylabel(r'$F(x)$')
        plt.grid(color='grey', linestyle=':')
        plt.legend()

        plt.show()

    # --------        fabric        --------
    @classmethod
    def from_shape(cls, shape: ApparatusShape) -> 'Apparatus':
        return cls(shape=shape)


if __name__ == '__main__':

    # aperture
    aperture = Apparatus(
        # shape=RectangularApparatusShape(width=25),
        shape=VoigtApparatusShape(width=25, asymmetry=0, ratio=.1),
    )
    aperture.show()
