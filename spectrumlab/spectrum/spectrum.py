import os
from typing import TypeAlias

import matplotlib as mpl
import matplotlib.pyplot as plt

from spectrumlab.alias import Array, NanoMeter, Number
from spectrumlab.emulation.spectrum import BaseSpectrum
from spectrumlab.emulation.detector.linear_array_detector import Detector
from spectrumlab.spectrum.base_spectrum import BaseSpectrum


def fetch_cmap(filename: str):

    filepath = os.path.join('.', 'colormaps', filename)
    with open(filepath, 'r') as file:
        lines = [list(map(float, line.strip().split(','))) for line in file.readlines()]

    cmap = mpl.colors.LinearSegmentedColormap.from_list('absorbance', lines)

    return cmap


# --------        spectrum        --------
class EmittedSpectrum(BaseSpectrum):
    """Type for any emitted (or ordinary) spectrum."""
    def __init__(self, intensity: Array[float], wavelength: Array[NanoMeter] | None = None, number: Array[Number] | None = None, deviation: Array[float] | None = None, clipped: Array[bool] | None = None, detector: Detector | None = None):
        super().__init__(intensity=intensity, wavelength=wavelength, number=number, clipped=clipped, deviation=deviation, detector=detector)

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


class AbsorbedSpectrum(BaseSpectrum):
    """Type for any absorbed spectrum."""
    def __init__(self, intensity: Array[float], wavelength: Array[NanoMeter] | None = None, number: Array[Number] | None = None, deviation: Array[float] | None = None, clipped: Array[bool] | None = None, detector: Detector | None = None):
        super().__init__(intensity=intensity, wavelength=wavelength, number=number, deviation=deviation, clipped=clipped, detector=detector)

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


# --------        assembly spectrum        --------
class AssemplySpectrum:
    """Type of spectrum from assemply device."""
    def __init__(self, items: tuple[Spectrum]):
        self.items = items

    # --------        handlers        --------
    def select(self, index: int) -> Spectrum:
        return self.items[index]
