from dataclasses import dataclass, field

import numpy as np
import pytest

from spectrumlab.alias import Array, MicroMeter
from spectrumlab.emulation.aperture import Aperture, RectangularApertureShape
from spectrumlab.emulation.apparatus import Apparatus, VoigtApparatusShape
from spectrumlab.emulation.detector import Detector
from spectrumlab.emulation.emulation import convolve
from spectrumlab.peak.shape import VoigtPeakShape


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
    ['detector', ],
    [
        (Detector.BLPP2000, ),
        (Detector.BLPP4000, ),
    ],
)
def test_shape(detector: Detector, config: Config):
    tolerance = 1e-4  # 0.01 [%]
    step = detector.config.width

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
        width=apparatus.shape.width/step,
        asymmetry=apparatus.shape.asymmetry,
        ratio=apparatus.shape.ratio,
        rx=int(config.rx/step),
        dx=config.dx/step,
    )
    f_hat = convolve(x/step, apparatus=apparatus, aperture=aperture, step=step)

    # 
    mask = (x > -(config.rx-apparatus.shape.width/2)) & (x < +(config.rx-apparatus.shape.width/2))  # remove edges
    error = (f(x/step, 0, 1) - f_hat(x/step)) / f(0, 0, 1)

    assert np.all(np.abs(error[mask]) < tolerance)


@pytest.mark.parametrize(
    ['detector', ],
    [
        (Detector.BLPP2000, ),
        (Detector.BLPP4000, ),
    ],
)
def test_integral(detector: Detector, config: Config):
    tolerance = 1e-4  # 0.01 [%]
    step = detector.config.width

    apparatus = Apparatus(
        detector=detector,
        shape=VoigtApparatusShape(config.width, config.asymmetry, config.ratio),
    )
    aperture = Aperture(
        detector=detector,
        shape=RectangularApertureShape(),
    )

    x = config.x
    f = convolve(x, apparatus=apparatus, aperture=aperture, step=step)

    integral = np.sum(f(x/step)) * (config.dx/step)
    assert np.abs(integral - 1) < tolerance


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    config = Config(
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

    step = detector.config.width

    x = config.x
    f = VoigtPeakShape(config.width/step, config.asymmetry, config.ratio)
    f_hat = convolve(x/step, apparatus=apparatus, aperture=aperture, step=step)

    #
    integral = np.sum(f_hat(x/step)) * (config.dx/step)
    print(f'intergal: {integral:.4f}')

    diff = np.max(np.abs(f_hat(x/step) - f(x/step, 0, 1)) / f(0, 0, 1))
    print(f'diff: {diff:.4f}')

    #
    y = f(x/step, 0, 1)
    plt.plot(
        x, y,
        color='red', linestyle='none', marker='s', markersize=3,
        label='$y$',
    )

    y_hat = f_hat(x/step)
    plt.plot(
        x, y_hat,
        color='black', linestyle='-', linewidth=1,
        label='$\hat{y}$',
    )

    plt.plot(
        x, y_hat - y,
        color='black', linestyle='none', marker='s', markersize=0.5,
        label='$error$',
    )

    plt.grid(color='grey', linestyle=':')
    plt.legend()
    plt.show()
