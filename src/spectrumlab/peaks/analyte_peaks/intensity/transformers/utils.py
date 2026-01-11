import numpy as np
import pandas as pd

from spectrumlab.types import Frame, R


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
        __data[~__data['mask']]['intensity'].min(),
        __data[~__data['mask']]['intensity'].max(),
    ])
