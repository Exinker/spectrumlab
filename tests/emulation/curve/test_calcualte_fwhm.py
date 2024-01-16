from dataclasses import dataclass, field
from functools import partial
from typing import Callable, Mapping

import numpy as np
import pytest

from spectrumlab.alias import Array
from spectrumlab.emulation.curve import gauss, estimate_fwhm


# --------        fixtures        --------
@dataclass
class Config:
    rx: float
    dx: float
    tolerance: float = field(default=1e-4)  # 0.01 [%]

    @property
    def x(self) -> Array[float]:
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
def test_estimate_fwhm_gauss(w: float, config: Config):
    x = config.x
    f = partial(gauss, x0=0, w=w)
    fwhm = 2*np.sqrt(2*np.log(2)) * w

    fwhm_hat = estimate_fwhm(x=x, y=f(x))
    assert np.abs((fwhm_hat - fwhm) / fwhm) < config.tolerance

