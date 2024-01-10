from dataclasses import dataclass, field
from typing import overload

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate, interpolate, signal

from spectrumlab.alias import Array, Number, MicroMeter, Percent
from spectrumlab.emulation.aperture import Aperture, RectangularApertureShape
from spectrumlab.emulation.detector import Detector
from spectrumlab.emulation.line import Line, PVoigtLineShape
from spectrumlab.picture.config import COLOR

# ----------------    peak    ----------------
@dataclass
class Peak:
    """Convolution of a line and a aperture shape shapes

    Author: Vaschenko Pavel
     Email: vaschenko@vmk.ru
      Date: 2013.04.12
    """
    line: Line
    aperture: Aperture
    dx: MicroMeter = field(default=0.01)  # шаг построения интерполяции
    rx: MicroMeter = field(default=100)  # границы построения интерполяции

    _x: Array[MicroMeter] = field(init=False, repr=False, default=None)
    _f: Array[Percent] = field(init=False, repr=False, default=None)

    def __post_init__(self):

        x = np.linspace(-self.rx, +self.rx, 2*int(self.rx/self.dx) + 1)

        f1 = lambda x: self.line(x, position=0, intensity=1)
        f2 = lambda x: self.aperture(x, n=0)
        f = signal.convolve(f1(x), f2(x), mode='same') * (x[-1] - x[0])/len(x)

        self._x = x
        self._f = f

    @overload
    def __call__(self, x: float, position: float, intensity: float) -> float: ...
    @overload
    def __call__(self, x: Array[float], position: float, intensity: float) -> Array[float]: ...
    def __call__(self, x, position, intensity):

        F = interpolate.interp1d(
            self._x,
            self._f,
            kind='linear',
            bounds_error=False,
            fill_value=0,
        )
        f = intensity*F(x - position)

        return f


@dataclass(frozen=True)
class _Peak:
    """Convolution (slow and accurate) of a line and a aperture shape shapes

    Author: Vaschenko Pavel
     Email: vaschenko@vmk.ru
      Date: 2022.06.01
    """
    line: Line
    aperture: Aperture
    rx: MicroMeter = field(default=100)  # границы построения интерполяции

    def __call__(self, number: Array[Number], position: float, intensity: float) -> Array[Number]:

        f = np.zeros(number.shape)
        for i, n in enumerate(number):

            f1 = lambda x: self.line(x, position, intensity)
            f2 = lambda x: self.aperture(x, n)

            value = integrate.quad(
                lambda x: f1(x) * f2(x),
                n - self.rx,
                n + self.rx,
            )[0]

            f[i] = value

        return f


def show_peak(peak: Peak | _Peak, position: MicroMeter, intensity: float, xscale: Number | MicroMeter = Number):

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4), tight_layout=True)

    #
    rx, dx = 100, .01
    x = np.linspace(position-rx, position+rx, 2*int(rx/dx) + 1)

    step = detector.config.width
    spam = rx // step
    wavelength = np.array([step*i for i in np.arange(-spam, spam+1, 1)])

    xvalues = x if xscale == MicroMeter else x/step
    yvalues = peak.line(x, position, intensity)
    plt.plot(
        xvalues, yvalues,
        color=COLOR['blue'],
        label=r'$I(x)$',
    )

    xvalues = wavelength if xscale == MicroMeter else wavelength/step
    yvalues = peak(wavelength, position, intensity)
    plt.fill_between(
        xvalues, yvalues,
        step='mid',
        facecolor=COLOR['pink'], edgecolor='k',
        alpha=0.2,
        label=r'$s_{k}$',
    )

    plt.title(f'{detector.config.name}')
    plt.xlabel(r'$x, \mu$' if xscale == MicroMeter else r'$k$')
    plt.ylabel('$s_{k}$, %')

    plt.grid(
        color='grey', linestyle=':',
    )
    plt.legend()

    plt.show()


if __name__ == '__main__':

    # detector
    detector = Detector.BLPP4000

    # line
    width, asymmetry, ratio = 20, 0, 0.1

    # peak
    position = 0
    intensity = 10

    peak = Peak(
        line=Line(
            shape=PVoigtLineShape(width, asymmetry, ratio),
        ),
        aperture=Aperture(
            shape=RectangularApertureShape(
                detector=detector,
            ),
        ),
    )
    # peak = _Peak(
    #     line=Line(
    #         shape=PVoigtLineShape(width, asymmetry, ratio),
    #     ),
    #     aperture=Aperture(
    #         shape=RectangularApertureShape(
    #             detector=detector,
    #         ),
    #     ),
    # )

    show_peak(peak, position, intensity, xscale=MicroMeter)
