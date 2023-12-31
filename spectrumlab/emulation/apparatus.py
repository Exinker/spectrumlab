from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

from spectrumlab.alias import Array, Micro
from spectrumlab.emulation.curve import rectangular, pvoigt
from spectrumlab.emulation.detector.linear_array_detector import Detector


# --------        shape        --------
@dataclass
class RectangularApparatusShape:
    """Rectangular device's apparatus shape."""
    width: Micro

    def __call__(self, x: Micro | Array[Micro], x0: Micro, step: float) -> Array[float]:
        f = rectangular(x/step, x0=x0/step, w=self.width/step)/step

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

    def __call__(self, x: Micro | Array[Micro], x0: Micro, step: float) -> Array[float]:
        f = pvoigt(x/step, x0=x0/step, w=self.width/step, a=self.asymmetry, r=self.ratio)/step

        return f


ApparatusShape = RectangularApparatusShape | TriangularApparatusShape | VoigtApparatusShape


# --------        apparatus        --------
@dataclass
class Apparatus:
    """
    Interface for any apparatus's profile.

    Author: Vaschenko Pavel
     Email: vaschenko@vmk.ru
      Date: 2014.03.24
    """
    detector: Detector
    shape: ApparatusShape

    @property
    def step(self) -> Micro:
        return self.detector.config.width

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

    # --------        private        --------
    def __call__(self, x: Micro | Array[Micro], x0: Micro) -> Array[float]:
        return self.shape(x, x0=x0, step=self.step)


if __name__ == '__main__':

    # detector
    detector = Detector.BLPP2000

    # apparatus
    apparatus = Apparatus(
        detector=detector,
        shape=VoigtApparatusShape(width=25, asymmetry=.3, ratio=.0),
    )
    apparatus.show()
