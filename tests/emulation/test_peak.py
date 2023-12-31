from dataclasses import dataclass, field

import numpy as np
import pytest
from scipy import interpolate, integrate, signal

from spectrumlab.alias import Array, Micro
from spectrumlab.emulation.aperture import Aperture, RectangularApertureShape
from spectrumlab.emulation.apparatus import Apparatus, VoigtApparatusShape
from spectrumlab.emulation.detector.linear_array_detector import Detector
from spectrumlab.peak.shape import VoightPeakShape


# --------        fixtures        --------
@dataclass
class Config:
    apparatus: Apparatus
    rx: Micro
    dx: Micro
    tolerance: float = field(default=1e-9)

    @property
    def x(self) -> Array[float]:
        return np.linspace(-self.rx, +self.rx, self.n)

    @property
    def n(self) -> int:
        return 2*self.rx*int(1/self.dx) + 1


@pytest.fixture
def detector() -> Detector:
    return 


@pytest.fixture
def config(detector: Detector) -> Config:
    return Config(
        apparatus=Apparatus(
            detector=detector,
            shape=VoigtApparatusShape(25, 0, 0.1),
        ),
        rx=140,
        dx=1e-4,
    )


def test_config(config: Config):
    assert len(config.x) == config.n


# --------        tests        --------
@pytest.mark.parametrize(
    ['detector', ],
    [
        (Detector.BLPP2000, ),
        (Detector.BLPP4000, ),
    ],
)
def test_voight_peak_shape(detector: Detector, config: Config):
    tolerance = 1e-9
    apparatus = config.apparatus
    aperture = Aperture(detector=detector, shape=RectangularApertureShape())

    step = detector.config.width

    # x
    x = config.x

    # f
    f = interpolate.interp1d(
        x,
        signal.convolve(apparatus(x, 0), aperture(x, 0), mode='same') * (x[-1] - x[0])/config.n,
        kind='linear',
        bounds_error=False,
        fill_value=0,
    )

    # f_hat
    f_hat = VoightPeakShape(apparatus.shape.width/step, apparatus.shape.asymmetry, apparatus.shape.ratio, rx=int(config.rx/step), dx=config.dx/step)

    # 
    mask = (x > -(config.rx-apparatus.shape.width/2)) & (x < +(config.rx-apparatus.shape.width/2))  # remove edges
    error = f(x) - f_hat(x/step, 0, 1)

    assert np.all(np.abs(error[mask]) < tolerance)


@pytest.mark.parametrize(
    ['detector', ],
    [
        (Detector.BLPP2000, ),
        (Detector.BLPP4000, ),
    ],
)
def test_voight_peak_integral(detector: Detector, config: Config):
    tolerance = 1e-2  # 1 [%]  FIXME: check it!
    apparatus = config.apparatus
    step = detector.config.width

    # x
    x = config.x

    # f
    f = VoightPeakShape(apparatus.shape.width/step, apparatus.shape.asymmetry, apparatus.shape.ratio, rx=10, dx=1e-4)

    #
    integral = integral = integrate.quad(
        lambda x: f(x, 0, 1),
        -10,
        +10,
    )[0]

    assert np.abs(integral - 1) < tolerance


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    tolerance = 1e-6
    detector = Detector.BLPP2000
    apparatus = Apparatus(shape=VoigtApparatusShape(25, .1, .1))
    aperture = Aperture(detector=detector, shape=RectangularApertureShape())

    step = detector.config.width

    rx = 100
    dx = 1e-3

    x = np.linspace(-rx, +rx, 2*rx*int(1/dx) + 1)
    f = interpolate.interp1d(
        x,
        signal.convolve(apparatus(x, 0), aperture(x, 0), mode='same') * (x[-1] - x[0])/len(x),
        kind='linear',
        bounds_error=False,
        fill_value=0,
    )

    f_hat = VoightPeakShape(apparatus.shape.width/step, apparatus.shape.asymmetry, apparatus.shape.ratio)

    y = f(x)
    plt.plot(
        x, y,
        color='black',
        label='$y$',
    )

    y_hat = f_hat(x/step, 0, 1)
    plt.plot(
        x, y_hat,
        linestyle='--',
        label='$\hat{y}$',
    )

    plt.plot(
        x, y_hat - y,
        color='black', linestyle='none', marker='s', markersize=0.5,
        label='$error$',
    )

    plt.grid(color='grey', linestyle=':'),
    plt.show()
