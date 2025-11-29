from typing import Callable, Self

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import interpolate

from spectrumlab.calibrators.concentration_calibrators.intensity_transformers.intensity_transformer import (
    AbstractIntensityTransformer,
)
from spectrumlab.types import C, Frame, R


def prepare_data(
    __data: Frame,
) -> Frame:

    data = pd.DataFrame(
        [
            {
                'probe': i,
                'parallel': j,
                'concentration': __data.loc[(i, j), 'concentration'],
                'intensity': np.nanmax(__data.loc[(i, j), 'intensity']),
            }
            for i, j in __data.index
        ],
        columns=['probe', 'parallel', 'concentration', 'intensity'],
    ).set_index(['probe', 'parallel'])

    data = data.dropna()
    data = data.groupby(level=0, sort=False).mean()

    return data


class RegressionIntensityTransformer(AbstractIntensityTransformer):

    @classmethod
    def create(
        cls,
        data: Frame,
        bounds: tuple[R, R],
        show: bool = False,
    ) -> Self:
        lb, ub = bounds

        mask = (lb <= data['intensity']) & (data['intensity'] <= ub)
        a = 1
        b = np.log10(np.array(data['intensity'][mask], dtype=float)).mean() - np.log10(np.array(data['concentration'][mask], dtype=float)).mean()
        p = np.array([a, b])
        intensity_predictor = lambda x: 10**(np.polyval(p, np.log10(x)))

        data['intensity_true'] = np.array(list(map(intensity_predictor, data['concentration'])))

        _data = data.copy()
        x = _data['concentration'].map(np.log10)
        y = _data['intensity'].map(np.log10)
        # mask = y >= x + b
        # y[mask] = x[mask] + b
        concentration_predictor = interpolate.interp1d(
            y, x,
            kind='linear',
            bounds_error=False,
            fill_value=np.nan,
        )

        if show:
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
            # plt.yscale('log')

            plt.grid(color='grey', linestyle=':')

            plt.show()

        return cls(
            concentration_predictor=concentration_predictor,
            intensity_predictor=intensity_predictor,
            bounds=(lb, ub),
        )

    def __init__(
        self,
        concentration_predictor: Callable[[R], C],
        intensity_predictor: Callable[[C], R],
        bounds: tuple[R, R],
    ) -> None:

        self.concentration_predictor = concentration_predictor
        self.intensity_predictor = intensity_predictor
        self.bounds = bounds

    def __call__(self, __value: R) -> R:

        if __value <= self.bounds[1]:
            return __value

        value = self.intensity_predictor(
            x=10**(self.concentration_predictor(np.log10(__value))),
        )
        return value
