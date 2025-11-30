from typing import Callable, Self

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import interpolate

from spectrumlab.calibrators.concentration_calibrators.intensity_transformers.intensity_transformer import (
    AbstractIntensityTransformer,
)
from spectrumlab.types import Array, C, Frame, R


class RegressionIntensityTransformer(AbstractIntensityTransformer):

    @classmethod
    def create(
        cls,
        frame: Frame,
        bounds: tuple[R, R],
        show: bool = False,
    ) -> Self:
        
        print(frame)

        if isinstance(frame.index, pd.MultiIndex):
            data = cls.process_frame(frame)
        else:
            data = frame.copy()

        print(data)

        # calculate params
        lb, ub = bounds
        mask = (lb <= data['intensity']) & (data['intensity'] <= ub)
        a = 1
        b = np.log10(np.array(data['intensity'][mask], dtype=float)).mean() - np.log10(np.array(data['concentration'][mask], dtype=float)).mean()
        params = (a, b)

        # transformer kernel
        x = np.log10(data['concentration'])
        y = np.log10(data['intensity'])
        print(x)
        print(y)
        kernel = interpolate.interp1d(
            y, x,
            kind='linear',
            bounds_error=False,
            fill_value=np.nan,
        )

        # create transformer
        transformer = cls(
            kernel=kernel,
            bounds=(lb, ub),
            params=params,
        )

        if show:
            data['intensity_true'] = transformer.estimate_intensity(data['concentration'])

            fig, (ax_left, ax_right) = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

            plt.sca(ax_left)
            plt.plot(
                data['concentration'],
                data['intensity'],
                color='grey', linestyle='none', marker='s', markersize=4,
            )

            plt.plot(
                data['concentration'][mask],
                data['intensity'][mask],
                color='black', linestyle='none', marker='s', markersize=4,
            )

            plt.plot(
                data['concentration'],
                data['intensity_true'],
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

            x = data['concentration']
            y = 100 * (data['intensity'] - data['intensity_true']) / data['intensity_true']
            plt.plot(
                x, y,
                color='grey', linestyle=':',
            )

            x = data['concentration']
            y = 100 * (data['intensity'] - data['intensity_true']) / data['intensity_true']
            plt.plot(
                x, y,
                color='grey', linestyle='none', marker='s', markersize=4,
            )

            x = data['concentration']
            y = 100 * (data['intensity'] - data['intensity_true']) / data['intensity_true']
            plt.plot(
                x[mask], y[mask],
                color='black', linestyle='none', marker='s', markersize=4,
            )

            plt.xlabel(r'$C$')
            plt.ylabel(r'$(R - \hat{R})/\hat{R}$, %')

            plt.xscale('log')

            plt.grid(color='grey', linestyle=':')

            plt.show()

        return transformer

    def __init__(
        self,
        bounds: tuple[R, R],
        params: tuple[float, float],
        kernel: Callable[[R], C],
    ) -> None:

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

    def __call__(self, __value: R) -> R:

        if __value < self.bounds[1]:
            return __value

        value = self.estimate_intensity(
            10**(self.kernel(np.log10(__value))),
        )
        print(__value, self.kernel(np.log10(__value)), value)
        return value

    @staticmethod
    def process_frame(
        frame: Frame,
    ) -> Frame:

        data = pd.DataFrame(
            [
                {
                    'probe': i,
                    'parallel': j,
                    'concentration': frame.loc[(i, j), 'concentration'],
                    'intensity': np.nanmax(frame.loc[(i, j), 'intensity']),
                }
                for i, j in frame.index
            ],
            columns=['probe', 'parallel', 'concentration', 'intensity'],
        ).set_index(['probe', 'parallel'])

        data = data.dropna(subset=['concentration'])
        data = data.groupby(level=0, sort=False).mean()

        return data
