
import os
from abc import abstractmethod
from typing import overload, TypeAlias

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from spectrumlab.alias import Array
from spectrumlab.emulation.spectrum import BaseSpectrum
from spectrumlab.emulation.detector.linear_array_detector import Detector


def fetch_cmap(filename: str):

    filepath = os.path.join('.', 'colormaps', filename)
    with open(filepath, 'r') as file:
        lines = [list(map(float, line.strip().split(','))) for line in file.readlines()]

    cmap = mpl.colors.LinearSegmentedColormap.from_list('absorbance', lines)

    return cmap


class RecordedSpectrum(BaseSpectrum):
    """Base type for any recorded spectrum."""
    def __init__(self, intensity: Array, wavelength: Array | None = None, number: Array | None = None, clipped: Array | None = None, crystal: Array | None = None, detector: Detector | None = None):
        deviation = np.zeros(intensity.shape)  # FIXME: refactor!

        super().__init__(intensity=intensity, deviation=deviation, wavelength=wavelength, number=number, clipped=clipped, detector=detector)

        self.crystal = crystal

    @overload
    def __getitem__(self, index: int | slice) -> 'Spectrum': ...
    """get spectrum at selected time or times"""
    @overload
    def __getitem__(self, index: tuple) -> 'Spectrum': ...
    """get spectrum at selected times and numbers"""
    def __getitem__(self, index):
        cls = self.__class__

        if isinstance(index, int | slice):
            time = index

            return cls(
                intensity=self.intensity[time],
                wavelength=self.wavelength,
                number=self.number,
                clipped=self.clipped[time],
                crystal=self.crystal,
                detector=self.detector,
            )

        if isinstance(index, tuple):
            time, number = index

            return cls(
                intensity=self.intensity[number] if self.intensity.ndim == 1 else self.intensity[time, number],
                wavelength=self.wavelength[number],
                number=self.number[number],
                clipped=self.clipped[number] if self.clipped.ndim == 1 else self.clipped[time, number],
                crystal=self.crystal[number],
                detector=self.detector,
            )

    def __add__(self, other: float | Array) -> 'Spectrum':
        cls = self.__class__

        return cls(
            intensity=self.intensity + other,
            wavelength=self.wavelength,
            number=self.number,
            clipped=self.clipped,
            crystal=self.crystal,
            detector=self.detector,
        )

    def __sub__(self, other: float | Array) -> 'Spectrum':
        cls = self.__class__

        return cls(
            intensity=self.intensity - other,
            wavelength=self.wavelength,
            number=self.number,
            clipped=self.clipped,
            crystal=self.crystal,
            detector=self.detector,
        )

    # --------        handlers        --------
    def select(self, *index) -> 'EmittedSpectrum':
        cls = self.__class__

        match index:
            case 'crystal', crystal:
                number = self.number[self.crystal == crystal]

            case 'number', n0, dn:
                number = self.number[n0-dn:n0+dn]

            case _:
                raise ValueError(f'index {index} is not supported!')

        return cls(
            intensity=self.intensity[number] if self.intensity.ndim == 1 else self.intensity[:, number],
            wavelength=self.wavelength[number],
            # number=self.number[number],    # TODO: don't remove! Reset a number values!
            clipped=self.clipped[number] if self.clipped.ndim == 1 else self.clipped[:, number],
            crystal=self.crystal[number],
            detector=self.detector,
        )

    # --------        handlers        --------
    @abstractmethod
    def show(self, canvas, yscale):
        pass


class EmittedSpectrum(RecordedSpectrum):
    """Type for any emitted (or ordinary) spectrum."""
    def __init__(self, intensity: Array, wavelength: Array | None = None, number: Array | None = None, clipped: Array | None = None, crystal: Array | None = None, detector: Detector | None = None):
        super().__init__(intensity=intensity, wavelength=wavelength, number=number, clipped=clipped, crystal=crystal, detector=detector)

    # --------        handlers        --------
    def show(self, ax: plt.Axes | None = None, figsize: tuple[float, float] = (6, 4), cmap=None, clim: tuple[float, float] | None = None, grid: bool = False) -> None:
        is_filling = ax is not None

        if not is_filling:
            fig, ax = plt.subplots(figsize=figsize, tight_layout=True)

        if self.n_times > 1:
            raise NotImplementedError

        else:
            x, y = self.wavelength, self.intensity
            ax.step(
                x, y,
                where='mid',
                color='black',
            )

            ax.set_xlabel('$\lambda, nm$')
            ax.set_ylabel('$I, \%$')

            if grid:
                ax.grid(color='grey', linestyle=':')

        if not is_filling:
            plt.show()


class AbsorbedSpectrum(RecordedSpectrum):
    """Type for any absorbed spectrum."""
    def __init__(self, intensity: Array, wavelength: Array | None = None, number: Array | None = None, clipped: Array | None = None, crystal: Array | None = None, detector: Detector | None = None):
        super().__init__(intensity=intensity, wavelength=wavelength, number=number, clipped=clipped, crystal=crystal, detector=detector)

    # --------        handlers        --------
    def show(self, ax: plt.Axes | None = None, figsize: tuple[float, float] = (6, 4), cmap=None, clim: tuple[float, float] | None = None) -> None:
        is_filling = ax is not None

        if not is_filling:
            fig, ax = plt.subplots(figsize=figsize, tight_layout=True)

        if cmap is None:
            cmap = fetch_cmap(filename='absorption 7.002.txt')

        if clim is None:
            clim = (-.01, .5)

        ax.imshow(
            self.intensity,
            origin='lower',
            cmap=cmap, clim=clim,
            aspect='auto',
        )
        ax.set_xticks(self.number[self.n_numbers//8::self.n_numbers//4])
        ax.set_xticklabels([f'{w:.3f}' for w in self.wavelength[self.n_numbers//8::self.n_numbers//4]])
        ax.set_yticks(self.time[::self.n_times//8])
        ax.set_yticklabels([f'{t}' for t in self.time[::self.n_times//8]])

        ax.set_xlabel('$\lambda, nm$')
        # ax.set_ylabel('$t, ms$')
        ax.set_ylabel('$time$')

        if not is_filling:
            plt.show()


Spectrum: TypeAlias = EmittedSpectrum | AbsorbedSpectrum
