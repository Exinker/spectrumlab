
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from spectrumlab.alias import Number

if TYPE_CHECKING:
    from spectrumlab.peak.analyte_peak import AnalytePeak


CORR_COEFF = {
    'Garanin': np.array([-1.592, 2.795, -1.003, 0.002]),
    'Vashchenko': np.array([-0.0110, 1.2814, -0.6216, 0.0010]),
}


@dataclass
class ParabolaPositionConfig:
    corr_coeff: Sequence[float] | None = field(default=None)
    verbose: bool = field(default=False)


def _correct_position(value: Number, coeff: Sequence[float]) -> Number:
    """Correct position value by table values.
    
    TODO: deviation calculations

    FIXME: fix self-absorption peak correction
    FIXME: fix clipped peak correction
    FIXME: autocalculate from peak shape
    """

    # calculate deviation
    deviation = value - round(value)

    # calculate correction
    if deviation <= 0:
        correction = +np.polyval(coeff, -deviation)
    else:
        correction = -np.polyval(coeff, +deviation)

    #
    return float(value + correction)


def estimate_position_by_parabola(peak: 'AnalytePeak', config: ParabolaPositionConfig) -> Number:
    """Estimate analyte peak's position by parabola approximation."""

    # check cursor on the edge
    if (peak.cursor == 0) or (peak.cursor == peak.n_numbers-1):
        return peak.number[peak.cursor]

    # estimate a position by parabola
    index = np.array([peak.cursor - 1, peak.cursor, peak.cursor + 1])
    x = peak.number[index]
    y = peak.value[index]

    Y = np.array([[y[0] - y[1]], [y[0] - y[2]]])
    M = np.array([
            [x[0]**2 - x[1]**2, x[0] - x[1]],
            [x[0]**2 - x[2]**2, x[0] - x[2]],
        ])
    X = np.linalg.solve(M, Y)

    value = float(-X[1] / 2 / X[0])

    # correct a position
    if config.corr_coeff is not None:
        value = _correct_position(value, coeff=config.corr_coeff)

    # verbose
    if config.verbose:
        print(f'Peak\'s position: {value}')

    #
    return value
