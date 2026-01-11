from typing import TypeAlias

from .absorbed_spectrum import AbsorbedSpectrum
# from .assembly_spectrum import AssemblySpectrum
from .emitted_spectrum import EmittedSpectrum
# from .high_resolution_spectrum import HighResolutionSpectrum


Spectrum: TypeAlias = EmittedSpectrum | AbsorbedSpectrum


__all__ = [
    AbsorbedSpectrum,
    # AssemblySpectrum,
    EmittedSpectrum,
    # HighResolutionSpectrum,
    Spectrum,
]
