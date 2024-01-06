from dataclasses import dataclass, field
from functools import partial

import numpy as np
import pytest

from spectrumlab.alias import Array, Micro
from spectrumlab.emulation.aperture import Aperture, ApertureShape, RectangularApertureShape
from spectrumlab.emulation.detector.linear_array_detector import Detector


@dataclass
class Config:
    rx: Micro = field(default=100)
    dx: Micro = field(default=1e-2)

    @property
    def x(self) -> Array[Micro]:
        return np.linspace(-self.rx, +self.rx, self.n)

    @property
    def n(self) -> int:
        return 2*int(self.rx/self.dx) + 1


# --------        fixtures        --------
@pytest.fixture
def config() -> Config:
    return Config()


# --------        tests        --------
@pytest.mark.parametrize(
    ['detector', 'shape'],
    [
        (Detector.BLPP2000, RectangularApertureShape()),
        (Detector.BLPP4000, RectangularApertureShape()),
    ]
)
def test_integral(detector: Detector, shape: ApertureShape, config: Config):
    tolerance = 1e-9

    x = config.x
    f = partial(Aperture(detector=detector, shape=shape), n=0)

    integral = np.sum(f(x))*config.dx
    assert np.abs(integral - 1) < tolerance
