import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from spectrumlab.peaks.analyte_peaks.intensity.transformers.regression_intensity_transformer.kernels.base_kernel import (
    KernelABC,
)
from spectrumlab.types import Array, C, R


class AmplitudeKernel(KernelABC):

    def __init__(
        self,
        intensity: Array[R],
        concentration: Array[C],
        bounds: tuple[R, R],
    ) -> None:
        super().__init__(intensity, concentration, bounds)

        x = np.log10(intensity)
        y = np.log10(concentration)

        order = np.argsort(y)

        interpolate = interp1d(
            x[order],
            y[order],
            kind='linear',
            bounds_error=False,
            fill_value=np.nan,
        )
       
        self.kernel = lambda value: 10**(np.polyval(
            p=(1, self.bias),
            x=float(interpolate(np.log10(value)).item()),
        ))
