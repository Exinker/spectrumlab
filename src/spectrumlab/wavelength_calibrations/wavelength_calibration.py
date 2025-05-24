from abc import ABC, abstractmethod
from typing import Callable

import numpy as np

from spectrumlab.spectra import Spectrum
from spectrumlab.types import Array, NanoMeter, Number
from spectrumlab.wavelength_calibrations.exceptions import FitError


class AbstractWavelengthCalibration(ABC):

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

    @abstractmethod
    def fit(self, number: Array[Number], wavelength: Array[NanoMeter]) -> 'AbstractWavelengthCalibration':
        raise NotImplementedError

    def predict(self, number: Number | Array[Number]) -> Array[NanoMeter]:
        return np.polyval(self.coeff, number)


class WavelengthCalibration(AbstractWavelengthCalibration):

    def __init__(self, deg: int) -> None:
        super().__init__(deg=deg)

    def fit(self, number: Array[Number], wavelength: Array[NanoMeter]) -> 'WavelengthCalibration':
        self._coeff = np.polyfit(number, wavelength, deg=self.deg)

        return self


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
