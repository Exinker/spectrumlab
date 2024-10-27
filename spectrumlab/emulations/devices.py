"""
Device for emulation.

Author: Vaschenko Pavel
 Email: vaschenko@vmk.ru
  Date: 2023.02.01
"""
from dataclasses import dataclass
from enum import Enum

from spectrumlab.types import MicroMeter, PicoMeter


@dataclass(frozen=True)
class DeviceConfig:
    """Device config."""
    name: str
    dispersion: float  # reciprocal linear dispersion [pm/μm]

    def __repr__(self) -> str:
        name = self.name
        dispersion = self.dispersion

        content = f'{type(self).__name__}: {name}\n\treciprocal linear dispersion: {dispersion:.4f} pm/μm'
        return content


@dataclass(frozen=False)
class DynamicDeviceConfig:
    """Device config (not frozen)."""
    name: str
    dispersion: float  # reciprocal linear dispersion [pm/μm]


class Device(Enum):
    """Enums with devices config."""

    COLIBRI2 = DeviceConfig(
        name='Колибри-2',
        dispersion=6.3857,
    )
    GRAND2_I = DeviceConfig(
        name='Гранд-2 (полихроматор I)',
        dispersion=0.4143,
    )
    GRAND2_II = DeviceConfig(
        name='Гранд-2 (полихроматор II)',
        dispersion=1.1286,
    )
    CONTRAA = DeviceConfig(
        name='contrAA® 800 D',
        dispersion=0.1,
    )
    dynamic = DynamicDeviceConfig(
        name='dynamic',
        dispersion=0,
    )

    @property
    def config(self) -> DeviceConfig | DynamicDeviceConfig:
        return self.value

    # --------        handlers        --------
    def estimate_resolution(self, slit_width: MicroMeter) -> PicoMeter:
        """Estimate device's resolution (theoretical limit)."""

        return self.config.dispersion * slit_width

    # --------        private        --------
    def __repr__(self) -> str:
        name = self.value.name
        dispersion = self.value.dispersion

        content = f'{type(self).__name__}: {name}\n\treciprocal linear dispersion: {dispersion:.4f} pm/μm'

        return content


if __name__ == '__main__':
    print(Device.GRAND2_I.value)
