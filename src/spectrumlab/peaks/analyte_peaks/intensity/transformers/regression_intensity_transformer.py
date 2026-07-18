from typing import Callable, Self

import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate

from spectrumlab.peaks.analyte_peaks.intensity.transformers.base_intensity_transformer import (
    IntensityTransformerABC,
)
from spectrumlab.types import Array, C, R


class RegressionIntensityTransformer(IntensityTransformerABC):

    @classmethod
    def create(
        cls,
        intensity: Array[R],
        concentration: Array[C],
        bounds: tuple[R, R],
    ) -> Self:

        intensity = np.array(intensity, dtype=np.float64, copy=True)
        concentration = np.array(concentration, dtype=np.float64, copy=True)

        if len(intensity) != len(concentration):
            raise ValueError('Intensity and concentration arrays must have the same length!')

        if np.any(intensity <= 0) or np.any(concentration <= 0):
            raise ValueError("Intensity and concentration arrays must contain strictly positive values!")

        # calculate params
        lb, ub = bounds
        mask = (lb <= intensity) & (intensity <= ub)

        if mask.sum() < 2:
            raise ValueError('The calibration bounds must contain at least 2 data points!')

        a = 1
        b = np.log10(intensity[mask]).mean() - np.log10(concentration[mask]).mean()  # noqa: E501
        params = (a, b)

        # transformer kernel
        x = np.log10(concentration)
        y = np.log10(intensity)
        order = np.argsort(y)

        interp = interpolate.interp1d(
            y[order],
            x[order],
            kind='linear',
            bounds_error=False,
            fill_value=np.nan,
        )

        kernel = lambda value: 10**(float(interp(np.log10(value)).item()))

        return cls(
            intensity=intensity,
            concentration=concentration,
            kernel=kernel,
            bounds=(lb, ub),
            params=params,
        )

    def __init__(
        self,
        intensity: Array[R],
        concentration: Array[C],
        bounds: tuple[R, R],
        params: tuple[float, float],
        kernel: Callable[[R], R],
    ) -> None:

        self.intensity = intensity
        self.concentration = concentration
        self.bounds = bounds
        self.params = params
        self.kernel = kernel

    def transform(self, __value: R) -> R:

        intensity_hat = 10**(np.polyval(self.params, np.log10(self.kernel(__value))))
        return intensity_hat

    def __call__(self, __value: R) -> R:

        if __value <= self.bounds[1]:
            return __value

        intensity_hat = self.transform(__value)
        return intensity_hat

    def estimate_intensity_hat(self, __value: C) -> R:

        intensity_hat = 10**(np.polyval(self.params, np.log10(__value)))
        return intensity_hat

    def show(self) -> None:
        lb, ub = self.bounds

        mask = (lb <= self.intensity) & (self.intensity <= ub)
        intensity_true = self.estimate_intensity_hat(self.concentration)

        fig, (ax_left, ax_right) = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

        plt.sca(ax_left)
        plt.plot(
            self.concentration,
            self.intensity,
            color='grey', linestyle='none', marker='s', markersize=4,
        )
        plt.plot(
            self.concentration[mask],
            self.intensity[mask],
            color='black', linestyle='none', marker='s', markersize=4,
        )
        plt.plot(
            self.concentration,
            intensity_true,
            color='grey', linestyle=':',
        )
        plt.axhspan(
            lb, ub,
            alpha=.125, color='red',
        )
        plt.xlabel('$C$')
        plt.ylabel('$R$')
        plt.xscale('log')
        plt.yscale('log')
        plt.grid(color='grey', linestyle=':')

        plt.sca(ax_right)
        x = self.concentration
        y = 100 * (self.intensity - intensity_true) / intensity_true
        plt.plot(
            x, y,
            color='grey', linestyle=':',
        )
        x = self.concentration
        y = 100 * (self.intensity - intensity_true) / intensity_true
        plt.plot(
            x, y,
            color='grey', linestyle='none', marker='s', markersize=4,
        )
        plt.plot(
            x[mask], y[mask],
            color='black', linestyle='none', marker='s', markersize=4,
        )
        plt.xlabel(r'$C$')
        plt.ylabel(r'$(R - \hat{R})/\hat{R}$, %')
        plt.xscale('log')
        plt.grid(color='grey', linestyle=':')

        plt.show()
