from dataclasses import dataclass, field
from functools import partial

import numpy as np
import pytest

from spectrumlab.alias import Array, Number
from spectrumlab.core.grid import Grid
from spectrumlab.core.grid.utils import estimate_fwhm
from spectrumlab.emulation.curve import gauss


# --------        fixtures        --------
@dataclass
class Config:
    rx: Number
    dx: Number
    tolerance: float = field(default=1e-4)  # 0.01 [%]

    @property
    def number(self) -> Array[Number]:
        return np.linspace(-self.rx, +self.rx, self.n)

    @property
    def n(self) -> int:
        return 2*int(self.rx/self.dx) + 1


@pytest.fixture
def config() -> Config:
    return Config(
        rx=10,
        dx=1e-2,
    )


# --------        tests        --------
@pytest.mark.parametrize(
    'w',
    [
        .5, 1, 2,
    ],
)
def test_estimate_fwhm_gauss(w: Number, config: Config):
    number = config.number
    f = partial(gauss, x0=0, w=w)
    fwhm = 2*np.sqrt(2*np.log(2)) * w

    fwhm_hat = estimate_fwhm(
        grid=Grid(x=number, y=f(number), units=Number),
        pitch=1,
    )
    assert np.abs((fwhm_hat - fwhm) / fwhm) < config.tolerance
