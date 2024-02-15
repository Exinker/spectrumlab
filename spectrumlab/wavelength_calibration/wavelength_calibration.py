from abc import ABC, abstractmethod
from typing import Callable

import numpy as np

from spectrumlab.alias import Array, NanoMeter, Number
from spectrumlab.spectrum import Spectrum
from spectrumlab.wavelength_calibration.exceptions import FitError


class BaseWavelengthCalibration(ABC):

    def __init__(self, deg: int) -> None:
        self._deg = deg

        self._coeff = None

    @property
    def deg(self) -> int:
        return self._deg

    @property
    def coeff(self) -> Array[float]:
        if self._coeff is None:
            raise FitError('Fit the calibration before!')

        return self._coeff

    # --------        handlers        --------
    @abstractmethod
    def fit(self, number: Array[Number], wavelength: Array[NanoMeter]) -> 'BaseWavelengthCalibration':
        raise NotImplementedError

    def predict(self, number: Number | Array[Number]) -> Array[NanoMeter]:
        return np.polyval(self.coeff, number)


class WavelengthCalibration(BaseWavelengthCalibration):

    def __init__(self, deg: int) -> None:
        super().__init__(deg=deg)

    # --------        handlers        --------
    def fit(self, number: Array[Number], wavelength: Array[NanoMeter]) -> 'WavelengthCalibration':
        self._coeff = np.polyfit(number, wavelength, deg=self.deg)

        return self


# --------        handlers        --------
def interpolate(spectrum: Spectrum, deg: int = 2) -> Callable[[Array[Number]], Array[NanoMeter]]:
    p = np.polyfit(spectrum.number, spectrum.wavelength, deg=deg)

    def inner(x: Array[Number]) -> Array[NanoMeter]:
        return np.polyval(p, x)

    return inner


def calibrate(spectrum: Spectrum, deg: int = 2) -> WavelengthCalibration:
    return WavelengthCalibration(
        deg=deg,
    ).fit(
        number=spectrum.number,
        wavelength=spectrum.wavelength,
    )
