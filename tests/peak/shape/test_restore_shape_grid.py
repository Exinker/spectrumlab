from dataclasses import dataclass, field

import numpy as np
import pytest

from spectrumlab.alias import Array, Number
from spectrumlab.core.grid import Grid
from spectrumlab.peak.shape import VoigtPeakShape, restore_shape_from_grid

from core import distance


@dataclass
class Config:
    width: Number
    asymmetry: float
    ratio: float
    rx: Number = field(default=10)
    dx: Number = field(default=1e-2)

    @property
    def number(self) -> Array[Number]:
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

    number = config.number
    shape = VoigtPeakShape(config.width, config.asymmetry, config.ratio)

    #
    grid = Grid(x=number, y=shape(number, 0, 1), units=Number)
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

    # shape
    config = Config(
        width=2.0,
        asymmetry=0.1,
        ratio=0.1,
    )
    shape = VoigtPeakShape(config.width, config.asymmetry, config.ratio)

    # shape_hat
    shape_hat = restore_shape_from_grid(
        grid=Grid(x=config.number, y=shape(config.number, 0, 1), units=Number),
        show=True,
    )
