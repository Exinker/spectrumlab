from dataclasses import dataclass, field

import numpy as np
import pytest

from spectrumlab.emulation.aperture import Aperture, RectangularApertureShape
from spectrumlab.emulation.apparatus import Apparatus, VoigtApparatusShape
from spectrumlab.emulation.detector import Detector
from spectrumlab.emulation.emulation import convolve
from spectrumlab.peak.shape import VoigtPeakShape
from spectrumlab.types import Array, MicroMeter


@dataclass
class Config:
    width: MicroMeter
    asymmetry: float
    ratio: float
    rx: MicroMeter = field(default=100)
    dx: MicroMeter = field(default=1e-2)
    tolerance: float = field(default=1e-6)

    @property
    def x(self) -> Array[float]:
        return np.linspace(-self.rx, +self.rx, self.n)

    @property
    def n(self) -> int:
        return 2*int(self.rx/self.dx) + 1


# --------        fixtures        --------
@pytest.fixture
def config() -> Config:
    return Config(
        width=28,
        asymmetry=0,
        ratio=0,
    )


# --------        tests        --------
@pytest.mark.parametrize(
    ['detector'],
    [
        (Detector.BLPP2000, ),
        (Detector.BLPP4000, ),
    ],
)
def test_shape(detector: Detector, config: Config):
    tolerance = 1e-4  # 0.01 [%]

    apparatus = Apparatus(
        detector=detector,
        shape=VoigtApparatusShape(config.width, config.asymmetry, config.ratio),
    )
    aperture = Aperture(
        detector=detector,
        shape=RectangularApertureShape(),
    )

    x = config.x
    f = VoigtPeakShape(
        width=apparatus.shape.width/detector.pitch,
        asymmetry=apparatus.shape.asymmetry,
        ratio=apparatus.shape.ratio,
        rx=int(config.rx/detector.pitch),
        dx=config.dx/detector.pitch,
    )
    f_hat = convolve(x/detector.pitch, apparatus=apparatus, aperture=aperture, pitch=detector.pitch)

    #
    mask = (x > -(config.rx-apparatus.shape.width/2)) & (x < +(config.rx-apparatus.shape.width/2))  # remove edges
    error = (f(x/detector.pitch, 0, 1) - f_hat(x/detector.pitch)) / f(0, 0, 1)

    assert np.all(np.abs(error[mask]) < tolerance)


@pytest.mark.parametrize(
    ['detector'],
    [
        (Detector.BLPP2000, ),
        (Detector.BLPP4000, ),
    ],
)
def test_integral(detector: Detector, config: Config):
    tolerance = 1e-4  # 0.01 [%]

    apparatus = Apparatus(
        detector=detector,
        shape=VoigtApparatusShape(config.width, config.asymmetry, config.ratio),
    )
    aperture = Aperture(
        detector=detector,
        shape=RectangularApertureShape(),
    )

    x = config.x
    f = convolve(x, apparatus=apparatus, aperture=aperture, pitch=detector.pitch)

    integral = np.sum(f(x/detector.pitch)) * (config.dx/detector.pitch)
    assert np.abs(integral - 1) < tolerance


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    config = Config(  # noqa: F811
        width=28,
        asymmetry=0,
        ratio=0,
    )

    detector = Detector.BLPP4000
    apparatus = Apparatus(
        detector=detector,
        shape=VoigtApparatusShape(config.width, config.asymmetry, config.ratio),
    )
    aperture = Aperture(
        detector=detector,
        shape=RectangularApertureShape(),
    )

    x = config.x
    f = VoigtPeakShape(config.width/detector.pitch, config.asymmetry, config.ratio)
    f_hat = convolve(x/detector.pitch, apparatus=apparatus, aperture=aperture, pitch=detector.pitch)

    #
    integral = np.sum(f_hat(x/detector.pitch)) * (config.dx/detector.pitch)
    print(f'intergal: {integral:.4f}')

    diff = np.max(np.abs(f_hat(x/detector.pitch) - f(x/detector.pitch, 0, 1)) / f(0, 0, 1))
    print(f'diff: {diff:.4f}')

    #
    y = f(x/detector.pitch, 0, 1)
    plt.plot(
        x, y,
        color='red', linestyle='none', marker='s', markersize=3,
        label='$y$',
    )

    y_hat = f_hat(x/detector.pitch)
    plt.plot(
        x, y_hat,
        color='black', linestyle='-', linewidth=1,
        label=fr'$\hat{y}$',
    )

    plt.plot(
        x, y_hat - y,
        color='black', linestyle='none', marker='s', markersize=0.5,
        label='$error$',
    )

    plt.grid(color='grey', linestyle=':')
    plt.legend()
    plt.show()
