import matplotlib.pyplot as plt

from spectrumlab.detectors import Detector
from spectrumlab.picture.colormaps import Colormap
from spectrumlab.spectra.base_spectrum import SpectrumABC
from spectrumlab.types import Array, NanoMeter, Number


class EmittedSpectrum(SpectrumABC):
    """Type for any emitted (or ordinary) spectrum."""
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
            clipped=clipped,
            deviation=deviation,
            detector=detector,
        )

    def show(
        self,
        ax: plt.Axes | None = None,
        figsize: tuple[float, float] = (6, 4),
        cmap: Colormap | None = None,
        clim: tuple[float, float] | None = None,
        grid: bool = False,
    ) -> None:
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

            ax.set_xlabel(r'$\lambda, nm$')
            ax.set_ylabel(r'$I, \%$')

            if grid:
                ax.grid(color='grey', linestyle=':')

        if not is_filling:
            plt.show()
