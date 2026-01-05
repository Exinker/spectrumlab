from abc import ABC, abstractmethod
from typing import Self, overload

import numpy as np

from spectrumlab.detectors import Detector
from spectrumlab.types import Array, NanoMeter, Number


@overload
def reshape(values: Array[float]) -> Array[float]: ...
@overload
def reshape(values: None) -> None: ...
def reshape(values):

    if values is None:
        return None

    if (values.ndim == 2) and (values.shape[0] == 1):
        return values.flatten()

    return values


class SpectrumABC(ABC):

    def __init__(
        self,
        intensity: Array[float],
        wavelength: Array[NanoMeter] | None = None,
        number: Array[Number] | None = None,
        deviation: Array[float] | None = None,
        clipped: Array[bool] | None = None,
        detector: Detector | None = None,
    ):
        self.intensity = reshape(intensity)
        self.detector = detector

        self._wavelength = reshape(wavelength)
        self._number = reshape(number)
        self._deviation = reshape(deviation)
        self._clipped = reshape(clipped)

        assert self.intensity.shape == self.clipped.shape
        assert self.intensity.shape == self.deviation.shape

    @property
    def n_times(self) -> int:
        if self.intensity.ndim == 1:
            return 1
        return self.intensity.shape[0]

    @property
    def time(self) -> Array[int]:
        return np.arange(self.n_times)

    @property
    def n_numbers(self) -> int:
        if self.intensity.ndim == 1:
            return self.intensity.shape[0]
        return self.intensity.shape[1]

    @property
    def index(self) -> Array[int]:
        """internal index of spectrum."""
        return np.arange(self.n_numbers)

    @property
    def number(self) -> Array[int]:
        """external index of spectrum."""
        if self._number is None:
            self._number = self.index

        return self._number

    @property
    def shape(self) -> tuple[int, int]:
        return self.intensity.shape

    @property
    def wavelength(self) -> Array[NanoMeter] | Array[Number]:
        if self._wavelength is None:
            self._wavelength = np.arange(self.n_numbers)

        return self._wavelength

    @property
    def deviation(self) -> Array[float]:
        if self._deviation is None:
            self._deviation = np.full(self.shape, 0)

        return self._deviation

    @property
    def clipped(self) -> Array[bool]:
        if self._clipped is None:
            self._clipped = np.full(self.shape, False)

        return self._clipped

    @abstractmethod
    def show(self, canvas, yscale):
        pass

    def __repr__(self) -> str:
        if self.intensity.ndim == 1:
            n_times, n_numbers = 1, self.shape[-1]
        else:
            n_times, n_numbers = self.shape

        cls = self.__class__
        return f'{cls.__name__}(n_times: {n_times}, n_numbers: {n_numbers})'

    @overload
    def __getitem__(self, index: int | slice) -> Self: ...
    """Get spectrum at selected time or times."""
    @overload
    def __getitem__(self, index: tuple[slice | Array[int], slice | Array[int]]) -> Self: ...
    """Get spectrum at selected times and numbers."""
    def __getitem__(self, index):
        cls = self.__class__

        if isinstance(index, int):
            """Select a frame of the spectrum by `index`."""
            assert self.n_times > 1, 'only time resolved spectra are supported!'

            time = index
            return cls(
                intensity=self.intensity[time],
                wavelength=self.wavelength,
                number=self.number,
                deviation=self.deviation[time],
                clipped=self.clipped[time],
                detector=self.detector,
            )

        if isinstance(index, slice | np.ndarray):
            """Select a frame or part of the spectrum by `index`."""
            if self.n_times > 1:
                time = index
                return cls(
                    intensity=self.intensity[time],
                    wavelength=self.wavelength,
                    number=self.number,
                    deviation=self.deviation[time],
                    clipped=self.clipped[time],
                    detector=self.detector,
                )

            else:
                number = index
                return cls(
                    intensity=self.intensity[number],
                    wavelength=self.wavelength[number],
                    number=self.number[number],
                    deviation=self.deviation[number],
                    clipped=self.clipped[number],
                    detector=self.detector,
                )

        if isinstance(index, tuple):
            time, number = index

            return cls(
                intensity=self.intensity[time, number],
                wavelength=self.wavelength[number],
                number=self.number[number],
                deviation=self.deviation[time, number],
                clipped=self.clipped[time, number],
                detector=self.detector,
            )

    def __add__(self, other: float | Array[float]) -> Self:
        cls = self.__class__

        return cls(
            intensity=self.intensity + other,
            wavelength=self.wavelength,
            number=self.number,
            deviation=self.deviation,
            clipped=self.clipped,
            detector=self.detector,
        )

    def __iadd__(self, other: float | Array[float]) -> Self:
        return self + other

    def __radd__(self, other: float | Array[float]) -> Self:
        return self + other

    def __sub__(self, other: float | Array[float]) -> Self:
        cls = self.__class__

        return cls(
            intensity=self.intensity - other,
            wavelength=self.wavelength,
            number=self.number,
            deviation=self.deviation,
            clipped=self.clipped,
            detector=self.detector,
        )

    def __isub__(self, other: float | Array[float]) -> Self:
        return self - other

    def __rsub__(self, other: float | Array[float]) -> Self:
        return self - other
