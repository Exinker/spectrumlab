from dataclasses import dataclass, field
from functools import partial

import numpy as np
import pytest

from spectrumlab.alias import Array, Micro
from spectrumlab.emulation.apparatus import Apparatus, ApparatusShape, VoigtApparatusShape
from spectrumlab.emulation.detector.linear_array_detector import Detector


# --------        fixtures        --------
@dataclass
class Config:
    rx: Micro
    dx: Micro
    tolerance: float = field(default=1e-2)

    @property
    def x(self) -> Array[Micro]:
        return np.linspace(-self.rx, +self.rx, self.n)

    @property
    def n(self) -> int:
        return 2*self.rx*int(1/self.dx) + 1


@pytest.fixture
def config() -> Config:
    return Config(
        rx=1000,
        dx=1e-2,
    )


@pytest.fixture
def detector() -> Detector:
    return 


def test_config(config: Config):
    assert len(config.x) == config.n


# --------        tests        --------
@pytest.mark.parametrize(
    ['detector', 'shape'],
    [
        (Detector.BLPP2000, VoigtApparatusShape(width=25, asymmetry=+0, ratio=0)),
        (Detector.BLPP4000, VoigtApparatusShape(width=25, asymmetry=+0, ratio=0)),
        (Detector.BLPP2000, VoigtApparatusShape(width=25, asymmetry=+0.1, ratio=0)),
    ]
)
def test_curve_integral(detector: Detector, shape: ApparatusShape, config: Config):
    x = config.x
    f = partial(Apparatus(detector=detector, shape=shape), x0=0)

    integral = np.sum(f(x))*config.dx
    assert np.abs(integral - 1) < config.tolerance
