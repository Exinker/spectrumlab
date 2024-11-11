"""
Emitted and absorbed spectrum emulation.

    Author: Vaschenko Pavel
    Email: vaschenko@vmk.ru
    Date: 2013.04.12

"""
from typing import TypeAlias, overload

from .base_emulator import AbstractEmulator
from .emitted_spectrum_emulator import EmittedSpectrumEmulator, EmittedSpectrumEmulationConfig, emitted_spectrum_factory, SpectrumConfig, convolve
from .high_dynamic_spectrum_emulation import HighDynamicRangeEmittedSpectrumEmulation, HighDynamicRangeMode, hdr_emitted_spectrum_factory
from .absorbed_spectrum_emulator import AbsorbedSpectrumEmulator, AbsorbedSpectrumEmulatorConfig, absorbed_spectrum_factory, SpectrumBaseConfig, calculate_absorbance


EmulationConfig: TypeAlias = EmittedSpectrumEmulationConfig | AbsorbedSpectrumEmulatorConfig
Emulation: TypeAlias = EmittedSpectrumEmulator | AbsorbedSpectrumEmulator


@overload
def fetch_emulation(
    config: EmittedSpectrumEmulationConfig,
) -> EmittedSpectrumEmulator: ...
@overload
def fetch_emulation(
    config: EmittedSpectrumEmulationConfig,
    mode: HighDynamicRangeMode,
) -> HighDynamicRangeEmittedSpectrumEmulation: ...
@overload
def fetch_emulation(
    config: AbsorbedSpectrumEmulatorConfig,
) -> AbsorbedSpectrumEmulator: ...
def fetch_emulation(config, mode=None):

    if isinstance(config, EmittedSpectrumEmulationConfig):
        if isinstance(mode, type(None)):
            return EmittedSpectrumEmulator(config=config)

        if isinstance(mode, HighDynamicRangeMode):
            return HighDynamicRangeEmittedSpectrumEmulation(config=config, mode=mode)

        raise TypeError()

    if isinstance(config, AbsorbedSpectrumEmulatorConfig):
        return AbsorbedSpectrumEmulator(config=config)

    raise TypeError()
