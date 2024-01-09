"""
Data types for emulation emission or absorbtion spectra.

Author: Vaschenko Pavel
 Email: vaschenko@vmk.ru
  Date: 2022.08.24
"""
from collections.abc import Mapping
from typing import overload, Literal, TypeAlias

import numpy as np
import matplotlib.pyplot as plt

from spectrumlab.alias import Array, Percent, Electron, Absorbance, MilliSecond, NanoMeter, Number
from spectrumlab.emulation.detector.linear_array_detector import Detector
from spectrumlab.spectrum.base_spectrum import BaseSpectrum


class EmittedSpectrum(BaseSpectrum):
    """Type for any emitted (or ordinary) spectrum."""
    def __init__(self, intensity: Array[float], wavelength: Array[NanoMeter] [NanoMeter] | None = None, number: Array[Number] | None = None, deviation: Array[float] | None = None, clipped: Array[bool] | None = None, detector: Detector | None = None):
        super().__init__(intensity=intensity, wavelength=wavelength, number=number, deviation=deviation, clipped=clipped, detector=detector)

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


class HighDynamicRangeEmittedSpectrum(EmittedSpectrum):
    """Type for any microwave or ICP spectrum."""

    def __init__(self, number: Array[Number], shorts: Mapping[MilliSecond, EmittedSpectrum], method: Literal['naive', 'weighted'], **kwargs):
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
    def __getitem__(self, index: int | slice) -> 'HighDynamicRangeEmittedSpectrum': ...
    """get spectrum at selected time or times"""
    @overload
    def __getitem__(self, index: tuple[slice | Array[int], slice | Array[int]]) -> 'HighDynamicRangeEmittedSpectrum': ...
    """get spectrum at selected times and numbers"""
    def __getitem__(self, index):
        raise NotImplementedError

    def __add__(self, other: float | Array[float]) -> 'HighDynamicRangeEmittedSpectrum':
        return NotImplemented

    def __sub__(self, other: float | Array[float]) -> 'HighDynamicRangeEmittedSpectrum':
        return NotImplemented


class AbsorbedSpectrum(BaseSpectrum):
    """Type for any absorbtion spectrum."""

    def __init__(self, intensity: Array[float], base: Array[float], wavelength: Array[NanoMeter] | None = None, number: Array[Number] | None = None, deviation: Array[float] | None = None, clipped: Array[bool] | None = None, detector: Detector | None = None):
        super().__init__(intensity=intensity, wavelength=wavelength, number=number, deviation=deviation, clipped=clipped, detector=detector)

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
    def __getitem__(self, index: int | slice) -> 'AbsorbedSpectrum': ...
    """get spectrum at selected time or times"""
    @overload
    def __getitem__(self, index: tuple[slice | Array[int], slice | Array[int]]) -> 'AbsorbedSpectrum': ...
    """get spectrum at selected times and numbers"""
    def __getitem__(self, index):
        raise NotImplementedError

    def __add__(self, other: float | Array[float]) -> 'AbsorbedSpectrum':
        return NotImplemented

    def __sub__(self, other: float | Array[float]) -> 'AbsorbedSpectrum':
        return NotImplemented


Spectrum: TypeAlias = EmittedSpectrum | HighDynamicRangeEmittedSpectrum | AbsorbedSpectrum
