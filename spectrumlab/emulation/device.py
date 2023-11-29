from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class DeviceConfig:
    '''Device config'''
    name: str
    dispersion: float  # reciprocal linear dispersion (in nm/mm) of nm

    def __repr__(self) -> str:
        name = self.name
        dispersion = self.dispersion

        content = f'{type(self).__name__}: {name}\n\treciprocal linear dispersion: {dispersion:.4f} nm/mm'

        return content


@dataclass(frozen=False)
class DynamicDeviceConfig:
    '''Dynamic Device config'''
    name: str
    dispersion: float  # reciprocal linear dispersion (in nm/mm) of nm


class Device(Enum):
    '''Enums with devices config'''

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
    contrAA = DeviceConfig(
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

    def __repr__(self) -> str:
        name = self.value.name
        dispersion = self.value.dispersion

        content = f'{type(self).__name__}: {name}\n\treciprocal linear dispersion: {dispersion:.4f} nm/mm'

        return content


if __name__ == '__main__':
    print(Device.GRAND2_I.value)
