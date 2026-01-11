from typing import Callable, Self

import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate

from spectrumlab.peaks.analyte_peaks.intensity.transformers.base_intensity_transformer import (
    IntensityTransformerABC,
)
from spectrumlab.types import Array, C, Frame, R


class RegressionIntensityTransformer(IntensityTransformerABC):

    @classmethod
    def create(
        cls,
        data: Frame,
        bounds: tuple[R, R],
    ) -> Self:

        # calculate params
        lb, ub = bounds
        mask = (lb <= data['intensity']) & (data['intensity'] <= ub)
        a = 1
        b = np.log10(np.array(data['intensity'][mask], dtype=float)).mean() - np.log10(np.array(data['concentration'][mask], dtype=float)).mean()  # noqa: E501
        params = (a, b)

        # transformer kernel
        x = data['concentration'].map(np.log10)
        y = data['intensity'].map(np.log10)
        kernel = interpolate.interp1d(
            y, x,
            kind='linear',
            bounds_error=False,
            fill_value=np.nan,
        )

        # create transformer
        transformer = cls(
            data=data,
            kernel=kernel,
            bounds=(lb, ub),
            params=params,
        )
        return transformer

    def __init__(
        self,
        data: Frame,
        bounds: tuple[R, R],
        params: tuple[float, float],
        kernel: Callable[[R], C],
    ) -> None:

        self.data = data
        self.bounds = bounds
        self.params = params
        self.kernel = kernel

    def estimate_intensity(
        self,
        __value: C,
    ) -> R:
        return 10**(np.polyval(self.params, np.log10(__value)))

    def apply(
        self,
        __value: Array[R],
    ) -> Array[R]:
        return np.array(list(map(self, __value)), dtype=float)

    def show(self) -> None:
        lb, ub = self.bounds

        mask = (lb <= self.data['intensity']) & (self.data['intensity'] <= ub)
        intensity_true = self.estimate_intensity(self.data['concentration'])

        fig, (ax_left, ax_right) = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

        plt.sca(ax_left)
        plt.plot(
            self.data['concentration'],
            self.data['intensity'],
            color='grey', linestyle='none', marker='s', markersize=4,
        )
        plt.plot(
            self.data['concentration'][mask],
            self.data['intensity'][mask],
            color='black', linestyle='none', marker='s', markersize=4,
        )
        plt.plot(
            self.data['concentration'],
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
        x = self.data['concentration']
        y = 100 * (self.data['intensity'] - intensity_true) / intensity_true
        plt.plot(
            x, y,
            color='grey', linestyle=':',
        )
        x = self.data['concentration']
        y = 100 * (self.data['intensity'] - intensity_true) / intensity_true
        plt.plot(
            x, y,
            color='grey', linestyle='none', marker='s', markersize=4,
        )
        x = self.data['concentration']
        y = 100 * (self.data['intensity'] - intensity_true) / intensity_true
        plt.plot(
            x[mask], y[mask],
            color='black', linestyle='none', marker='s', markersize=4,
        )
        plt.xlabel(r'$C$')
        plt.ylabel(r'$(R - \hat{R})/\hat{R}$, %')
        plt.xscale('log')
        plt.grid(color='grey', linestyle=':')

        plt.show()

    def __call__(self, __value: R) -> R:

        if __value <= self.bounds[1]:
            return __value

        value = self.estimate_intensity(
            10**(self.kernel(np.log10(__value))),
        )
        return value
