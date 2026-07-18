import numpy as np
import pandas as pd
from scipy import interpolate

from spectrumlab.peaks.analyte_peaks.intensity.transformers.regression_intensity_transformer.kernels.base_kernel import (
    KernelABC,
)
from spectrumlab.types import C, R


class AmplitudeKernel(KernelABC):

    def __init__(
        self,
        intensity: pd.Series[R],
        concentration: pd.Series[C],
        bounds: tuple[R, R],
    ) -> None:
        super().__init__(intensity, concentration, bounds)

        x = np.log10(concentration)
        y = np.log10(intensity)
        order = np.argsort(y)

        f = interpolate.interp1d(
            y[order],
            x[order],
            kind='linear',
            bounds_error=False,
            fill_value=np.nan,
        )        
        self.kernel = lambda value: 10**(np.polyval(
            p=(1, self.bias),
            x=float(f(np.log10(value)).item()),  # FIXME: зачем эти преобразования во float?
        ))
