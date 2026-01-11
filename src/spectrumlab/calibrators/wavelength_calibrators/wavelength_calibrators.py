from abc import ABC, abstractmethod
from typing import Callable, Self

import numpy as np

from spectrumlab.calibrators.wavelength_calibrators.exceptions import WavelengthCalibratorError
from spectrumlab.spectra import Spectrum
from spectrumlab.types import Array, NanoMeter, Number


class WavelengthCalibratorABC(ABC):

    @abstractmethod
    def fit(self, number: Array[Number], wavelength: Array[NanoMeter]) -> Self:
        raise NotImplementedError

    @abstractmethod
    def predict(self, number: Number | Array[Number]) -> Array[NanoMeter]:
        raise NotImplementedError


class RegressionWavelengthCalibrator(WavelengthCalibratorABC):

    def __init__(self, deg: int) -> None:

        self._deg = deg

        self._coeff = None

    @property
    def deg(self) -> int:
        return self._deg

    @property
    def coeff(self) -> Array[float]:
        if self._coeff is None:
            raise WavelengthCalibratorError('Fit the wavelength calibrator before!')

        return self._coeff

    def fit(self, number: Array[Number], wavelength: Array[NanoMeter]) -> Self:
        self._coeff = np.polyfit(number, wavelength, deg=self.deg)

        return self

    def predict(self, number: Number | Array[Number]) -> Array[NanoMeter]:
        return np.polyval(self.coeff, number)


def interpolate(
    spectrum: Spectrum,
    calibrator: WavelengthCalibratorABC | None = None,
) -> Callable[[Array[Number]], Array[NanoMeter]]:

    calibrator = calibrator or RegressionWavelengthCalibrator(deg=2)
    calibrator = calibrator.fit(
        number=spectrum.number,
        wavelength=spectrum.wavelength,
    )

    def inner(__number: Array[Number]) -> Array[NanoMeter]:
        return calibrator.predict(__number)

    return inner


def calibrate(
    spectrum: Spectrum,
    calibrator: WavelengthCalibratorABC | None = None,
) -> RegressionWavelengthCalibrator:

    calibrator = calibrator or RegressionWavelengthCalibrator(deg=2)
    calibrator = calibrator.fit(
        number=spectrum.number,
        wavelength=spectrum.wavelength,
    )

    return calibrator
