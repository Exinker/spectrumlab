from dataclasses import dataclass, field

import numpy as np
import pytest

from spectrumlab.alias import Array, MicroMeter
from spectrumlab.emulation.detector import Detector
from spectrumlab.peak.shape import VoightPeakShape


@dataclass
class Config:
    width: MicroMeter
    asymmetry: float
    ratio: float
    rx: MicroMeter = field(default=100)
    dx: MicroMeter = field(default=1e-2)

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
def test_integral(detector: Detector, config: Config):
    tolerance = 1e-4  # 0.01 [%]
    step = detector.config.width

    x = config.x
    f = VoightPeakShape(config.width/step, config.asymmetry, config.ratio, rx=config.rx/step, dx=config.dx/step)

    integral = np.sum(f(x/step, 0, 1)) * (config.dx/step)
    assert np.abs(integral - 1) < tolerance


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    config = Config(
        width=28,
        asymmetry=0.1,
        ratio=0.1,
    )
    detector = Detector.BLPP2000
    step = detector.config.width

    x = config.x
    # f = VoightPeakShape(config.width/step, config.asymmetry, config.ratio, rx=config.rx/step, dx=config.dx/step)
    f = VoightPeakShape(config.width/step, config.asymmetry, config.ratio)

    # shape
    from spectrumlab.core.grid import Grid

    # grid = Grid(x/step, f(x/step, 0, 1))
    # grid.show()

    VoightPeakShape.from_grid(
        grid=Grid(x/step, f(x/step, 0, 1)),
        show=True,
    )

    # integral
    integral = np.sum(f(x/step, 0, 1)) * (config.dx/step)
    print(f'intergal: {integral:.4f}')

    #
    y = f(x/step, 0, 1)
    plt.plot(
        x, y,
        color='red', linestyle='none', marker='s', markersize=3,
        label='$y$',
    )

    plt.grid(color='grey', linestyle=':')
    plt.legend()
    plt.show()
