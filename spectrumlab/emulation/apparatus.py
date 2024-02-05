from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

from spectrumlab.alias import Array, MicroMeter
from spectrumlab.emulation.curve import rectangular, pvoigt
from spectrumlab.emulation.detector import Detector


# --------        shape        --------
@dataclass
class RectangularApparatusShape:
    """Rectangular device's apparatus shape."""
    width: MicroMeter

    def __call__(self, x: MicroMeter | Array[MicroMeter], x0: MicroMeter, step: MicroMeter) -> Array[float]:
        f = rectangular(x/step, x0=x0/step, w=self.width/step)/step

        return f


@dataclass
class TriangularApparatusShape:
    """Triangular device's apparatus shape."""
    width: MicroMeter

    def __call__(self, x: MicroMeter | Array[MicroMeter], x0: MicroMeter) -> Array[float]:
        raise NotImplementedError


@dataclass 
class VoigtApparatusShape:
    """Voigt device's apparatus shape.

    A simple asymmetric line shape profile for fitting infrared absorption spectra.
    Aaron L. Stancik, Eric B. Brauns
    https://www.sciencedirect.com/science/article/abs/pii/S0924203108000453
    """
    width: MicroMeter
    asymmetry: float
    ratio: float

    def __call__(self, x: MicroMeter | Array[MicroMeter], x0: MicroMeter, pitch: MicroMeter) -> Array[float]:
        f = pvoigt(x/pitch, x0=x0/pitch, w=self.width/pitch, a=self.asymmetry, r=self.ratio)/pitch

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
    def pitch(self) -> MicroMeter:
        return self.detector.pitch

    # --------        handlers        --------
    def show(self, rx: MicroMeter = 100, dx: MicroMeter = .01) -> None:
        
        #
        fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)

        x = np.linspace(-rx, +rx, 2*int(rx/dx) + 1)
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
    def __call__(self, x: MicroMeter | Array[MicroMeter], x0: MicroMeter = 0) -> Array[float]:
        return self.shape(x, x0=x0, pitch=self.pitch)


if __name__ == '__main__':

    # detector
    detector = Detector.BLPP2000

    # apparatus
    apparatus = Apparatus(
        detector=detector,
        shape=VoigtApparatusShape(width=25, asymmetry=0.0, ratio=0.0),
    )
    apparatus.show()

    # 
    rx = 100
    dx = 1e-2
    x = np.linspace(-rx, +rx, 2*int(rx/dx))

    integral = np.sum(apparatus(x, x0=0))*dx
    print(f'integral: {integral:.4f}')

    # 
    rx = 10
    dx = 1e-2/detector.pitch
    x = np.linspace(-rx, +rx, 2*int(rx/dx))
    integral = np.sum(apparatus(detector.pitch*x, x0=0))*(detector.pitch*dx)
    print(f'integral: {integral:.4f}')
