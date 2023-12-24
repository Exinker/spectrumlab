
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from spectrumlab.utils import mse
from spectrumlab.peak.shape import PeakShape, VoightPeakShape, EffectedVoightPeakShape

if TYPE_CHECKING:
    from spectrumlab.peak.analyte_peak import AnalytePeak


@dataclass
class Variables:

    peak: 'AnalytePeak'
    shape: PeakShape

    delta: float = field(default=1)
    by_tail: bool = field(default=False)

    _initial: tuple[float] = field(init=False, repr=False)  # initial values
    _bounds: tuple[tuple[float, float]] = field(init=False, repr=False)  # bounds of values
    _keys: tuple[str] = field(init=False, repr=False)  # keys
    _values: tuple[float] = field(init=False, repr=False)  # values

    def __post_init__(self):
        shape = self.shape

        if isinstance(shape, VoightPeakShape):
            self._init_voight_shape()

        if isinstance(shape, EffectedVoightPeakShape):
            self._init_effected_voight_shape()

    def _init_voight_shape(self):
        peak = self.peak
        config = peak.settings.intensityConfig

        # shape
        emission_shape = config.approx_shape.emission_shape

        background = 0
        position = peak.position
        delta = self.delta + 1e-10
        intensity = approx_peak_by_tail(
            peak=peak,
            shape=emission_shape,
        )

        self._keys = ('background', 'position', 'intensity')
        self._initial = (background, position, intensity)
        self._bounds = (
            (-1e-10, +1e-10),  # FIXME: нужно разобраться с пределами у фона!
            (position - delta, position + delta),
            (0, np.inf),
        )

    def _init_effected_voight_shape(self):
        peak = self.peak

        config = peak.settings.intensityConfig

        # shape
        emission_shape = config.approx_shape.emission_shape
        absorption_shape = config.approx_shape.absorption_shape

        # initial
        background = 0
        position = peak.position
        delta = self.delta + 1e-10
        intensity = approx_peak_by_tail(
            peak=peak,
            shape=emission_shape,
        )
        absorption = 0

        # 
        to_restrict = False  # np.mean(peak.value[peak.index[peak.mask]]) < 2  # FIXME: restrict constant

        if absorption_shape is None:
            self._keys = ('background', 'position', 'intensity', 'absorption', 'width_absorption', 'ratio_absorption')
            self._initial = (background, position, intensity, absorption, emission_shape.width, emission_shape.ratio)
            self._bounds = (
                (-1e-10, +1e-10),  # FIXME: нужно разобраться с пределами у фона!
                (position - delta, position + delta),
                (0, np.inf),
                (0, 0 if to_restrict else np.inf),
                (1, 20),
                (0, 1),
            )

        else:
            self._keys = ('background', 'position', 'intensity', 'absorption')
            self._initial = (background, position, intensity, absorption)
            self._bounds = (
                (-1e-10, +1e-10),  # FIXME: нужно разобраться с пределами у фона!
                (position - delta, position + delta),
                (0, np.inf),
                (0, 0 if to_restrict else np.inf),
            )

    @property
    def index(self):
        peak = self.peak

        if self.by_tail:
            index = peak.tail
        else:
            index = peak.index

        return index[peak.mask[index]]

    @property
    def x(self):
        index = self.index
        peak = self.peak

        return peak.number[index]

    @property
    def y(self):
        index = self.index
        peak = self.peak

        return peak.value[index]

    def keys(self) -> tuple[str]:
        """keys of variables"""
        return self._keys

    @property
    def initial(self) -> tuple:
        """initial guess of values of variables"""

        return self._initial

    def get_initial(self) -> dict:
        """initial guess of values of variables"""
        return {
            key: value
            for key, value in zip(self._keys, self._initial)
        }

    @property
    def bounds(self):
        """bounds on variables"""
        return self._bounds

    def get_values(self) -> dict:
        """values of variables"""
        return {
            key: value
            for key, value in zip(self._keys, self._values)
        }

    def get_value(self, key: str) -> float:
        values = self.get_values()

        return values.get(key, None)

    def set_values(self, values: tuple) -> None:
        self._values = values

    def parse_values(self, values: tuple) -> dict:
        return {
            key: value
            for key, value in zip(self._keys, values)
        }


def approx_peak_by_tail(peak: 'AnalytePeak', shape: PeakShape) -> float:
    """Approximate a analyte peak with selected shape on the tail"""

    # index
    index = peak.tail
    index = index[peak.mask[index]]

    # intensity
    x = peak.number[index]
    y = peak.value[index]
    y_hat = shape(x=x, **{'position': peak.position, 'intensity': 1})

    intensity = np.dot(y,y) / np.dot(y_hat,y)

    #
    return intensity


def approx_peak(peak: 'AnalytePeak', shape: PeakShape, delta: float = 1, by_tail: bool = False, show: bool = False) -> dict:
    """Approximate a analyte peak with selected shape."""

    def _fitness(params: Sequence[float], shape: PeakShape, variables: Variables) -> float:
        """Interface to calculate a fitness of approximation of a analyte peak by any shape"""
        x, y = variables.x, variables.y

        return mse(
            y=y,
            y_hat=shape(x=x, **variables.parse_values(values=params)),
        )

    #
    variables = Variables(peak=peak, shape=shape, delta=delta, by_tail=by_tail)

    result = minimize(
        partial(_fitness, shape=shape, variables=variables),
        variables.initial,
        method='SLSQP',
        bounds=variables.bounds,
    )
    values = result.x

    variables.set_values(values)

    # draw
    if show:
        x, y = peak.number, peak.value
        plt.plot(x, y, color='black', marker='s', markersize=.5, alpha=.2)

        x, y = peak.number, peak.value
        y[~peak.mask] = np.nan
        plt.plot(x, y, marker='s', markersize=.5)

        left, right = peak.minima
        x = np.linspace(left, right, 1000)
        y_hat = shape(x, **variables.get_values())
        plt.plot(x, y_hat, color='red')

        x, y = peak.number, peak.value
        y_hat = shape(x, **variables.get_values())
        plt.plot(x, y - y_hat, color='black', linestyle=':')

        plt.grid()
        plt.show()

    return variables.get_values()


@dataclass
class ApproxIntensityConfig:
    approx_shape: PeakShape
    approx_params: dict = field(default_factory=dict)
    delta: float = field(default=1)  # deviation of peak's position
    by_tail: bool = field(default=False)  # use tail for approximation

    @property
    def color(self) -> str:
        return '#9467bd'


def estimate_intensity_by_approx(peak: 'AnalytePeak', config: ApproxIntensityConfig, verbose: bool = False, show: bool = False) -> float:
    """Estimate analyte peak's intensity by approximation."""

    # approx
    if not config.approx_params:  # FIXME (2023): это нужно?
        config.approx_params = approx_peak(
            peak=peak,
            shape=config.approx_shape,
            delta=config.delta,
            by_tail=config.by_tail,
            show=show,
        )

    value = config.approx_params['intensity']

    # verbose
    if verbose:
        print(f'Peak\'s intensity: {value}')

    #
    return value