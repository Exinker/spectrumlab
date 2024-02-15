from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import pytest

from spectrumlab.alias import Array, Number
from spectrumlab.peak.shape.voigt_peak_shape import SelfReversedVoigtPeakShapeNaive


@dataclass
class Config:
    shape: SelfReversedVoigtPeakShapeNaive
    rx: Number = field(default=10)
    dx: Number = field(default=1e-2)

    @property
    def n(self) -> int:
        return 2*int(self.rx/self.dx) + 1

    @property
    def x(self) -> Array[Number]:
        return np.linspace(-self.rx, +self.rx, self.n)


@pytest.mark.parametrize('config', [
    Config(
        shape=SelfReversedVoigtPeakShapeNaive(
            width=25/14,
            asymmetry=0,
            ratio=0.1,
        ),
    ),
])
def test_shape_error(config: Config):
    tolerance = 1e-3

    for effect in np.linspace(0, 4, 1001):
        y = config.shape._apply_effect(config.x, effect=effect)
        y_hat = config.shape(config.x, position=0, intensity=1, effect=effect)

        assert np.all(np.abs(y_hat - y) < tolerance)


if __name__ == '__main__':
    config = Config(
        shape=SelfReversedVoigtPeakShapeNaive(
            width=25/14,
            asymmetry=0,
            ratio=0.1,
        ),
    )
    re = 4
    de = .25
    effect = np.linspace(0, re, int(re/de))  # remove `+1` because of intergrid

    error = []
    for xi in effect:
        x = config.x
        y = config.shape._apply_effect(x, effect=xi)
        plt.plot(
            x, y,
            color='black', linestyle='-', linewidth=1,
            label=fr'$\xi$: {xi:.1f}',
        )

        y_hat = config.shape(x, position=0, intensity=1, effect=xi)
        plt.plot(
            x, y_hat,
            color='red', linestyle='-', linewidth=1,
        )

        error.append(np.max(np.abs(y_hat - y)))
    error = np.array(error)

    plt.xlabel(r'number')
    plt.ylabel(r'$I$ [$\%$]')
    plt.grid(color='grey', linestyle=':')
    plt.legend()
    plt.show()

    plt.plot(effect, error)
    plt.grid(color='grey', linestyle=':')
    plt.show()
