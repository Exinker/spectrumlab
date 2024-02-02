from typing import Callable

import numpy as np

from spectrumlab.alias import Array, Number, NanoMeter
from spectrumlab.spectrum import Spectrum


def interpolate(spectrum: Spectrum, deg: int = 2) -> Callable[[Array[Number]], Array[NanoMeter]]:
    p = np.polyfit(spectrum.number, spectrum.wavelength, deg=deg)

    def inner(x: Array[Number]) -> Array[NanoMeter]:
        return np.polyval(p, x)

    return inner
