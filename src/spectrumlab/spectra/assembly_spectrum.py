from spectrumlab.spectra.base_spectrum import SpectrumABC


class AssemblySpectrum:
    """Type of spectrum from assemply device."""
    def __init__(self, items: tuple[SpectrumABC, ...]):
        self.items = items

    def select(self, index: int) -> SpectrumABC:
        return self.items[index]
