"""
Data types for emulation emission or absorbtion spectra.

Author: Vaschenko Pavel
 Email: vaschenko@vmk.ru
  Date: 2022.08.24
"""
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import overload, Literal, TypeAlias

import numpy as np
import matplotlib.pyplot as plt

from spectrumlab.alias import Array, Percent, Electron, Absorbance, MilliSecond
from spectrumlab.emulation.detector.linear_array_detector import Detector


@overload
def reshape(values: Array[float]) -> Array[float]: ...
@overload
def reshape(values: None) -> None: ...
def reshape(values):  

    if values is None:
        return None

    if (values.ndim == 2) and (values.shape[0] == 1):
        return values.reshape(-1, )

    return values


# ----------------    spectrum    ----------------
class BaseSpectrum(ABC):
    """Base type for any spectrum."""

    def __init__(self, intensity: Array, deviation: Array, wavelength: Array | None = None, number: Array | None = None, clipped: Array | None = None, detector: Detector | None = None):

        self.intensity = reshape(intensity)
        self.deviation = reshape(deviation)
        self.detector = detector

        self._wavelength = reshape(wavelength)
        self._number = reshape(number)
        self._clipped = reshape(clipped)

        assert self.intensity.shape == self.clipped.shape

    @property
    def n_times(self):
        if self.intensity.ndim == 1:
            return 1
        return self.intensity.shape[0]

    @property
    def time(self):
        return np.arange(self.n_times)

    @property
    def n_numbers(self):
        if self.intensity.ndim == 1:
            return self.intensity.shape[0]
        return self.intensity.shape[1]

    @property
    def index(self):
        """internal index of spectrum."""
        return np.arange(self.n_numbers)

    @property
    def number(self):
        """external index of spectrum."""
        if self._number is None:
            self._number = self.index

        return self._number

    @property
    def shape(self):
        return self.intensity.shape

    @property
    def wavelength(self):
        if self._wavelength is None:
            self._wavelength = np.arange(self.n_numbers)

        return self._wavelength

    @property
    def clipped(self):
        if self._clipped is None:
            self._clipped = np.full(self.shape, False)

        return self._clipped

    # --------        private        --------
    def __repr__(self):
        if self.intensity.ndim == 1:
            n_times, n_numbers = 1, self.shape[-1]
        else:
            n_times, n_numbers = self.shape

        cls = self.__class__
        return f'{cls.__name__}(n_times: {n_times}, n_numbers: {n_numbers})'

    @overload
    def __getitem__(self, index: int | slice): ...
    """Get spectrum at selected time or times."""
    @overload
    def __getitem__(self, index: tuple[slice | Array, slice | Array]): ...
    """Get spectrum at selected times and numbers."""
    def __getitem__(self, index):
        cls = self.__class__

        if isinstance(index, int | slice):
            time = index

            return cls(
                intensity=self.intensity[time],
                deviation=self.deviation,
                wavelength=self.wavelength,
                number=self.number,
                clipped=self.clipped[time],
                detector=self.detector,
            )

        if isinstance(index, tuple):
            time, number = index

            return cls(
                intensity=self.intensity[time, number],
                deviation=self.deviation[number],
                wavelength=self.wavelength[number],
                number=self.number[number],
                clipped=self.clipped[time, number],
                detector=self.detector,
            )

    def __add__(self, other: float | Array):
        cls = self.__class__

        return cls(
            intensity=self.intensity + other,
            deviation=self.deviation,
            wavelength=self.wavelength,
            number=self.number,
            clipped=self.clipped,
            detector=self.detector,
        )

    def __iadd__(self, other: float | Array):
        return self + other

    def __radd__(self, other: float | Array):
        return self + other

    def __sub__(self, other: float | Array):
        cls = self.__class__

        return cls(
            intensity=self.intensity - other,
            deviation=self.deviation,
            wavelength=self.wavelength,
            number=self.number,
            clipped=self.clipped,
            detector=self.detector,
        )

    def __isub__(self, other: float | Array):
        return self - other

    def __rsub__(self, other: float | Array):
        return self - other

    # --------        handlers        --------
    @abstractmethod
    def show(self, canvas, yscale):
        pass


class EmittedSpectrum(BaseSpectrum):
    """Type for any emitted (or ordinary) spectrum."""
    def __init__(self, intensity: Array, deviation: Array, wavelength: Array | None = None, number: Array | None = None, clipped: Array | None = None, detector: Detector | None = None):
        super().__init__(intensity=intensity, deviation=deviation, wavelength=wavelength, number=number, clipped=clipped, detector=detector)

    # --------        handlers        --------
    def show(self, ax: plt.Axes | None = None, figsize: tuple[float, float] = (6, 4), yscale: Percent | Electron = Percent) -> None:
        is_filling = ax is not None

        if not is_filling:
            fig, ax = plt.subplots(figsize=figsize, tight_layout=True)

        # draw integral spectrum
        x = self.wavelength
        y = self.intensity
        ax.step(
            x, y,
            where='mid',
            color='black',
        )

        # set axes
        ax.set_xlabel(r'$\lambda$ [$nm$]')
        ax.set_ylabel(
            r'$I$ [$\bar{e}$]' if yscale is Electron else r'$I$ [$\%$]'
        )

        ax.grid(
            color='grey', linestyle=':',
        )

        if not is_filling:
            plt.show()

    # --------        private        --------
    def __add__(self, other: float | Array):
        cls = self.__class__

        return cls(
            intensity=self.intensity + other,
            deviation=self.deviation,
            wavelength=self.wavelength,
            number=self.number,
            clipped=self.clipped,
            detector=self.detector,
        )

    def __sub__(self, other: float | Array):
        cls = self.__class__

        return cls(
            intensity=self.intensity - other,
            deviation=self.deviation,
            wavelength=self.wavelength,
            number=self.number,
            clipped=self.clipped,
            detector=self.detector,
        )


class HighDynamicRangeEmittedSpectrum(EmittedSpectrum):
    """Type for any microwave or ICP spectrum."""

    def __init__(self, number: Array, shorts: Mapping[MilliSecond, EmittedSpectrum], method: Literal['naive', 'weighted'], **kwargs):
        n_shorts = len(shorts)
        n_numbers = len(number)

        match method:
            case 'naive':
                counts = np.zeros((n_numbers, ))
                intensity = np.zeros((n_numbers, ))
                variance = np.zeros((n_numbers, ))
                for n in number:

                    for tau, spe in shorts.items():
                        if not spe.clipped[n]:
                            counts[n] += 1
                            intensity[n] += spe.intensity[n] / tau
                            variance[n] += (spe.deviation[n] / tau) ** 2

                intensity = intensity / counts
                deviation = np.sqrt(variance) / counts
                clipped = np.min([spe.clipped for tau, spe in shorts.items()], axis=0)

            case 'weighted':
                intensity = np.zeros((n_shorts, n_numbers))
                deviation = np.zeros((n_shorts, n_numbers))
                weight = np.zeros((n_shorts, n_numbers))
                for i, (tau, spe) in enumerate(shorts.items()):
                    intensity[i] = spe.intensity / tau
                    deviation[i] = spe.deviation / tau
                    deviation[i][spe.clipped] = np.infty
                    weight[i] = (1 / deviation[i]) ** 2

                intensity = np.array([np.dot(intensity, w) for intensity, w in zip(intensity.T, weight.T)] / np.sum(weight, axis=0))
                deviation = np.sqrt(np.array([np.dot(deviation[deviation < np.infty]**2, w[deviation < np.infty]**2) for deviation, w in zip(deviation.T, weight.T)] / np.sum(weight, axis=0) ** 2))
                clipped = np.min([spe.clipped for tau, spe in shorts.items()], axis=0)

            case _:
                raise ValueError(f'method {method} is not supported!')

        # restore clipped values
        if any(clipped):
            intensity[clipped] = [spe.intensity / tau for tau, spe in shorts.items()][-1][clipped]
            deviation[clipped] = [spe.deviation / tau for tau, spe in shorts.items()][-1][clipped]

        #
        super().__init__(
            intensity=intensity,
            deviation=deviation,
            clipped=clipped,
            **kwargs,
        )

        #
        self.shorts = shorts

    # --------        private        --------
    @overload
    def __getitem__(self, index: int | slice): ...
    """get spectrum at selected time or times"""
    @overload
    def __getitem__(self, index: tuple[slice | Array, slice | Array]): ...
    """get spectrum at selected times and numbers"""
    def __getitem__(self, index):
        raise NotImplementedError

    def __add__(self, other: float | Array):
        return NotImplemented

    def __sub__(self, other: float | Array):
        return NotImplemented


class AbsorbedSpectrum(BaseSpectrum):
    """Type for any absorbtion spectrum."""

    def __init__(self, intensity: Array, deviation: Array, base: Array, wavelength: Array | None = None, number: Array | None = None, clipped: Array | None = None, detector: Detector | None = None):
        super().__init__(intensity=intensity, deviation=deviation, wavelength=wavelength, number=number, clipped=clipped, detector=detector)

        self.base = base

    # --------        handlers        --------
    def show(self, ax: plt.Axes | None = None, figsize: tuple[float, float] = (6, 4), yscale: Absorbance = Absorbance) -> None:
        is_filling = ax is not None

        if not is_filling:
            fig, ax = plt.subplots(figsize=figsize, tight_layout=True)

        # draw integral spectrum
        x = self.wavelength
        y = self.intensity
        ax.step(
            x, y,
            where='mid',
            color='black',
        )

        # set axes
        ax.set_xlabel(r'$\lambda$ [$nm$]')
        ax.set_ylabel(r'$A$')

        ax.grid(
            color='grey', linestyle=':',
        )

        if not is_filling:
            plt.show()

    # --------        private        --------
    @overload
    def __getitem__(self, index: int | slice): ...
    """get spectrum at selected time or times"""
    @overload
    def __getitem__(self, index: tuple[slice | Array, slice | Array]): ...
    """get spectrum at selected times and numbers"""
    def __getitem__(self, index):
        raise NotImplementedError

    def __add__(self, other: float | Array):
        return NotImplemented

    def __sub__(self, other: float | Array):
        return NotImplemented


Spectrum: TypeAlias = EmittedSpectrum | HighDynamicRangeEmittedSpectrum | AbsorbedSpectrum
