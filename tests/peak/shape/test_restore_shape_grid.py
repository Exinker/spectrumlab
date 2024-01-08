from dataclasses import dataclass, field

import numpy as np
import pytest

from spectrumlab.alias import Array, Number
from spectrumlab.peak.shape import VoightPeakShape, Grid, restore_shape_from_grid

from core import distance


@dataclass
class Config:
    width: Number
    asymmetry: float
    ratio: float
    rx: Number = field(default=10)
    dx: Number = field(default=1e-2)

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
        width=2,
        asymmetry=+0.1,
        ratio=0.1,
    )


# --------        tests        --------
def test_restore_shape_from_grid(config: Config):
    tolerance = 1e-4

    x = config.x
    shape = VoightPeakShape(config.width, config.asymmetry, config.ratio)

    #
    grid = Grid(x, shape(x, 0, 1))
    shape_hat = restore_shape_from_grid(
        grid=grid,
        show=False,
    )

    assert distance(
        xi=shape.width,
        xi_hat=shape_hat.width,
    ) < tolerance
    assert distance(
        xi=shape.asymmetry,
        xi_hat=shape_hat.asymmetry,
    ) < tolerance
    assert distance(
        xi=shape.ratio,
        xi_hat=shape_hat.ratio,
    ) < tolerance


if __name__ == '__main__':
    config = Config(
        width=2.0,
        asymmetry=0.1,
        ratio=0.1,
    )

    # shape
    x = config.x
    shape = VoightPeakShape(config.width, config.asymmetry, config.ratio)

    # shape_hat
    shape_hat = restore_shape_from_grid(
        grid=Grid(x, shape(x, 0, 1)),
        show=True,
    )
