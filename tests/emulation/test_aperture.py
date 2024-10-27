from dataclasses import dataclass, field
from functools import partial

import numpy as np
import pytest

from spectrumlab.types import Array, MicroMeter
from spectrumlab.emulations.apertures import Aperture, ApertureShape, RectangularApertureShape   # noqa: I100
from spectrumlab.emulations.detectors import Detector


@dataclass
class Config:
    rx: MicroMeter = field(default=100)
    dx: MicroMeter = field(default=1e-2)

    @property
    def x(self) -> Array[MicroMeter]:
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
    ],
)
def test_integral(detector: Detector, shape: ApertureShape, config: Config):
    tolerance = 1e-9

    x = config.x
    f = partial(Aperture(detector=detector, shape=shape), n=0)

    integral = np.sum(f(x))*config.dx
    assert np.abs(integral - 1) < tolerance
