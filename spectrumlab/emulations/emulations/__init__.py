"""
Emitted and absorbed spectrum emulation.

    Author: Vaschenko Pavel
    Email: vaschenko@vmk.ru
    Date: 2013.04.12

"""
from typing import TypeAlias, overload

from .emulation import AbstractEmulation
from .emitted_spectrum_emulation import EmittedSpectrumEmulation, EmittedSpectrumEmulationConfig, emulate_emitted_spectrum, SpectrumConfig, convolve
from .high_dynamic_spectrum_emulation import HighDynamicRangeEmittedSpectrumEmulation, HighDynamicRangeMode, emulate_hdr_emitted_spectrum
from .absorbed_spectrum_emulation import AbsorbedSpectrumEmulation, AbsorbedSpectrumEmulationConfig, emulate_absorbed_spectrum, SpectrumBaseConfig, calculate_absorbance


EmulationConfig: TypeAlias = EmittedSpectrumEmulationConfig | AbsorbedSpectrumEmulationConfig
Emulation: TypeAlias = EmittedSpectrumEmulation | AbsorbedSpectrumEmulation


# --------        handlers        --------
@overload
def fetch_emulation(config: EmittedSpectrumEmulationConfig) -> EmittedSpectrumEmulation: ...
@overload
def fetch_emulation(config: EmittedSpectrumEmulationConfig, mode: HighDynamicRangeMode) -> HighDynamicRangeEmittedSpectrumEmulation: ...
@overload
def fetch_emulation(config: AbsorbedSpectrumEmulationConfig) -> AbsorbedSpectrumEmulation: ...
def fetch_emulation(config, mode=None):

    if isinstance(config, EmittedSpectrumEmulationConfig):
        if isinstance(mode, type(None)):
            return EmittedSpectrumEmulation(config=config)

        if isinstance(mode, HighDynamicRangeMode):
            return HighDynamicRangeEmittedSpectrumEmulation(config=config, mode=mode)

        raise TypeError()

    if isinstance(config, AbsorbedSpectrumEmulationConfig):
        return AbsorbedSpectrumEmulation(config=config)

    raise TypeError()
