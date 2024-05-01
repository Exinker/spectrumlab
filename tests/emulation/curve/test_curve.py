from dataclasses import dataclass, field
from functools import partial
from typing import Callable, Mapping

import numpy as np
import pytest

from spectrumlab.emulation.curve import gauss, pvoigt, rectangular
from spectrumlab.types import Array


# --------        fixtures        --------
@dataclass
class Config:
    rx: float
    dx: float
    tolerance: float = field(default=1e-9)

    @property
    def x(self) -> Array[float]:
        return np.linspace(-self.rx, +self.rx, self.n)

    @property
    def n(self) -> int:
        return 2*int(self.rx/self.dx) + 1


@pytest.fixture
def config() -> Config:
    return Config(
        rx=50,
        dx=1e-4,
    )


# --------        tests        --------
@pytest.mark.parametrize(
    ['curve', 'params'],
    [
        (gauss, dict(x0=0, w=2)),
        (pvoigt, dict(x0=0, w=2, a=0, r=0)),
        (pvoigt, dict(x0=0, w=2, a=0, r=1)),
        (pvoigt, dict(x0=0, w=2, a=-0.1, r=0)),
        (pvoigt, dict(x0=0, w=2, a=-0.2, r=1)),
        (rectangular, dict(x0=0, w=1)),
    ],
)
def test_curve_flip(curve: Callable, params: Mapping[str, float], config: Config):
    x = config.x
    f = partial(curve, **params)

    def flip(x) -> Callable:
        if curve == pvoigt:
            return partial(curve, x0=params['x0'], w=params['w'], a=-params['a'], r=params['r'])(-x)

        return partial(curve, **params)(-x)

    assert np.all(np.abs(f(x) - flip(x)) < config.tolerance)


@pytest.mark.parametrize(
    ['curve', 'params'],
    [
        (gauss, dict(x0=0, w=2)),
        (pvoigt, dict(x0=0, w=2, a=0, r=0)),
        # (pvoigt, dict(x0=0, w=2, a=0, r=.1)),  # FIXME: check it!
        # (pvoigt, dict(x0=0, w=20, a=0, r=.1)),  # FIXME: check it!
        # (pvoigt, dict(x0=0, w=2, a=-0.1, r=0)),  # FIXME: check it!
        # (pvoigt, dict(x0=0, w=2, a=+0.1, r=0)),  # FIXME: check it!
        (rectangular, dict(x0=0, w=1)),
    ],
)
def test_curve_integral(curve: Callable, params: Mapping[str, float], config: Config):
    x = config.x
    f = partial(curve, **params)

    integral = np.sum(f(x))*config.dx
    assert np.abs(integral - 1) < config.tolerance
