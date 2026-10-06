from functools import partial

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import minimize

from spectrumlab.peaks.analyte_peaks.intensity.transformers.regression_intensity_transformer.kernels.base_kernel import (
    KernelABC,
)
from spectrumlab.types import Array, C, R


class IntegralKernel(KernelABC):

    def __init__(
        self,
        intensity: Array[R],
        concentration: Array[C],
        value: 'pd.Series[Array[R]]',
        bounds: tuple[R, R],
        n: int = 10,
        alpha: float = 1e-5,
        slope: float | None = 1.0,
    ) -> None:
        super().__init__(intensity, concentration, bounds)

        x = np.linspace(bounds[1], value.map(np.nanmax).max(), n)

        result = minimize(
            partial(
                loss,
                x=x,
                concentration=concentration,
                value=value,
                alpha=alpha,
                slope=slope,
            ),
            x0=np.zeros(n),
            bounds=[
                (0, None)
                for _ in range(n)
            ],
            method='L-BFGS-B',
        )
        if not result['success']:
            print(result)

        self.kernel = interp1d(
            x=x,
            y=x + np.cumsum(result['x']),
            kind='linear',
            bounds_error=False,
            fill_value=np.nan,
        )


def loss(
    delta: Array[R],
    x: Array[R],
    concentration: Array[C],
    value: 'pd.Series[Array[R]]',
    alpha: float,
    slope: float | None,
) -> float:
    y = x + np.cumsum(delta)

    # estimate intensity
    intensity = value.apply(partial(
        estimate_intensity,
        x=x,
        y=y,
    ))
    intensity = intensity.groupby(level=0, sort=False).mean()

    # calculate loss
    error = estimate_error(
        concentration=concentration,
        intensity=intensity,
        slope=slope,
    )
    penalty = alpha*np.sum(np.abs(delta))

    loss = error + penalty
    return loss


def estimate_intensity(
    value: Array[R],
    x: Array[R],
    y: Array[R],
) -> Array[R]:

    calibrate = interp1d(
        x, y,
        kind='linear',
        bounds_error=False,
    )

    value_hat = calibrate(value)
    value_hat = np.where(value < np.min(x), value, value_hat)
    return np.nansum(value_hat)


def estimate_error(
    concentration: Array[C],
    intensity: Array[R],
    slope: float | None,
) -> float:
    mask = ((concentration > 0) & ~np.isnan(concentration)) & ((intensity > 0) & ~np.isnan(intensity))
    if not np.any(mask):
        return float('nan')

    x = np.log10(concentration[mask])
    y = np.log10(intensity[mask])

    if slope is None:
        slope, bias = np.polyfit(x, y, 1)

    else:
        bias = np.nanmean(y - slope*x)

    y_hat = slope*x + bias
    return np.linalg.norm(y - y_hat)
