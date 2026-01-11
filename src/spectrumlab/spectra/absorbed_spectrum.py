import matplotlib.pyplot as plt

from spectrumlab.detectors import Detector
from spectrumlab.picture.colormaps import Colormap, fetch_cmap
from spectrumlab.spectra.base_spectrum import SpectrumABC
from spectrumlab.types import Array, NanoMeter, Number


class AbsorbedSpectrum(SpectrumABC):
    """Type for any absorbed spectrum."""
    def __init__(
        self,
        intensity: Array[float],
        wavelength: Array[NanoMeter] | None = None,
        number: Array[Number] | None = None,
        deviation: Array[float] | None = None,
        clipped: Array[bool] | None = None,
        detector: Detector | None = None,
    ) -> None:
        super().__init__(
            intensity=intensity,
            wavelength=wavelength,
            number=number,
            deviation=deviation,
            clipped=clipped,
            detector=detector,
        )

    def show(
        self,
        ax: plt.Axes | None = None,
        figsize: tuple[float, float] = (6, 4),
        cmap: Colormap | None = None,
        clim: tuple[float, float] | None = None,
    ) -> None:
        is_filling = ax is not None
        cmap = cmap or fetch_cmap(kind='absorption')
        clim = clim or (-.01, .5)

        #
        if not is_filling:
            fig, ax = plt.subplots(figsize=figsize, tight_layout=True)

        img = ax.imshow(
            self.intensity,
            origin='lower',
            cmap=cmap, clim=clim,
            aspect='auto',
        )

        ax.set_xticks(self.index[self.n_numbers//8::self.n_numbers//4])
        ax.set_xticklabels([f'{w:.3f}' for w in self.wavelength[self.n_numbers//8::self.n_numbers//4]])
        ax.set_yticks(self.time[::self.n_times//8])
        ax.set_yticklabels([f'{t}' for t in self.time[::self.n_times//8]])

        ax.set_xlabel(r'$\lambda$ [$nm$]')
        ax.set_ylabel(r'time')

        plt.colorbar(img, cmap=cmap)

        if not is_filling:
            plt.show()
