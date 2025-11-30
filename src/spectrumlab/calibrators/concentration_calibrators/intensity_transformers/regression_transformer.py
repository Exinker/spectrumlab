from typing import Callable, Self

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import interpolate

from spectrumlab.calibrators.concentration_calibrators.intensity_transformers.intensity_transformer import (
    AbstractIntensityTransformer,
)
from spectrumlab.types import Array, C, Frame, R


def process_frame(
    __frame: Frame,
) -> Frame:

    data = pd.DataFrame(
        [
            {
                'probe': i,
                'parallel': j,
                'concentration': __frame.loc[(i, j), 'concentration'],
                'intensity': np.nanmax(__frame.loc[(i, j), 'intensity']),
            }
            for i, j in __frame.index
        ],
        columns=['probe', 'parallel', 'concentration', 'intensity'],
    ).set_index(['probe', 'parallel'])

    data = data.dropna(subset=['concentration'])
    data = data.groupby(level=0, sort=False).mean()

    return data


def estimate_bounds(
    __data: Frame,
    threshold=0.05,
) -> tuple[R, R]:

    __data['mask'] = False

    x = __data['concentration'].map(np.log10)
    y = __data['intensity'].map(np.log10)

    intercept, slope = 0, 1
    while len(__data[~__data['mask']].index) > 2:
        values = y[~__data['mask']] - x[~__data['mask']]
        intercept, slope = np.mean(values), 1

        #
        i_true = 10**(intercept + slope*x)
        i_hat = 10**(y)
        if np.max((np.abs(i_true - i_hat) / i_true)[~__data['mask']]) > threshold:
            __data.loc[__data[~__data['mask']]['concentration'].idxmax(), 'mask'] = True  # mask the last of unmasked!
        else:
            break

    return tuple([
        __data[~__data['mask']]['intensity'].min().item(),
        __data[~__data['mask']]['intensity'].max().item(),
    ])


class RegressionIntensityTransformer(AbstractIntensityTransformer):

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
        b = np.log10(np.array(data['intensity'][mask], dtype=float)).mean() - np.log10(np.array(data['concentration'][mask], dtype=float)).mean()
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

        if __value < self.bounds[1]:
            return __value

        value = self.estimate_intensity(
            10**(self.kernel(np.log10(__value))),
        )
        return value
