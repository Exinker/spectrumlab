import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np

from spectrumlab.emulations.photodiodes.characteristics.characteristic import (
    AbstractCharacteristic,
    ConstantCharacteristic,
    DatasheetCharacteristic,
)
from spectrumlab.types import Array, Meter, NanoMeter


DATASHEET_DIRECTORY = os.path.join(os.path.dirname(__file__), 'datasheet')


@dataclass(frozen=True)
class DetectorConfig:
    """Detector's config."""
    name: str
    sensitivity: AbstractCharacteristic
    transmittance: AbstractCharacteristic
    description: str = field(default='')

    def __repr__(self) -> str:
        cls = self.__class__
        name = self.name

        return '\n'.join([
            f'{cls.__name__}: {name}',
            f'\n\n{self.description}' if self.description else '',
        ])


class Detector(Enum):
    """Enums with detector's config."""
    unicorn = DetectorConfig(
        name='unicorn',
        sensitivity=ConstantCharacteristic(value=1),
        transmittance=ConstantCharacteristic(value=1),
        description='Idealized detector example.',
    )
    G12180 = DetectorConfig(
        name='G12180 series',
        sensitivity=DatasheetCharacteristic.create(
            path=os.path.join(DATASHEET_DIRECTORY, 'G12180', 'photo-sensitivity.csv'),
            xscale=1e+6,
        ),
        transmittance=DatasheetCharacteristic.create(
            path=os.path.join(DATASHEET_DIRECTORY, 'G12180', 'window-spectral-transmittance.csv'),
            xscale=1e+6,
            yscale=100,
        ),
        description=r"""See more info at [hamamatsu](https://www.hamamatsu.com/content/dam/hamamatsu-photonics/sites/documents/99_SALES_LIBRARY/ssd/g12180_series_kird1121e.pdf)""",  # noqa: E501
    )
    G12183 = DetectorConfig(
        name='G12183 series*',  # exclude G12183-219KA-03 Detector
        sensitivity=DatasheetCharacteristic.create(
            path=os.path.join(DATASHEET_DIRECTORY, 'G12183', 'photo-sensitivity.csv'),
            xscale=1e+6,
        ),
        transmittance=DatasheetCharacteristic.create(
            path=os.path.join(DATASHEET_DIRECTORY, 'G12183', 'window-spectral-transmittance.csv'),
            xscale=1e+6,
            yscale=100,
        ),
        description=r"""See more info at [hamamatsu](https://www.hamamatsu.com/content/dam/hamamatsu-photonics/sites/documents/99_SALES_LIBRARY/ssd/g12183_series_kird1119e.pdf)""",  # noqa: E501
    )
    G12183_219KA_03 = DetectorConfig(
        name='G12183-219KA-03',
        sensitivity=DatasheetCharacteristic.create(
            path=os.path.join(DATASHEET_DIRECTORY, 'G12183-219KA-03', 'photo-sensitivity.csv'),
            xscale=1e+6,
        ),
        transmittance=DatasheetCharacteristic.create(
            path=os.path.join(DATASHEET_DIRECTORY, 'G12183-219KA-03', 'window-spectral-transmittance.csv'),
            xscale=1e+6,
            yscale=100,
        ),
        description=r"""See more info at [hamamatsu](https://www.hamamatsu.com/content/dam/hamamatsu-photonics/sites/documents/99_SALES_LIBRARY/ssd/g12183_series_kird1119e.pdf)""",  # noqa: E501
    )

    @property
    def config(self) -> DetectorConfig:
        return self.value

    def responce(self, x: Array[Meter], fill_value: float = np.nan) -> Array[float]:
        config = self.config

        return config.sensitivity(x, fill_value) * config.transmittance(x, fill_value)

    def show(
        self,
        bounds: tuple[NanoMeter, NanoMeter],
        info: Literal['title', 'text', 'none'] = 'text',
        save: bool = False,
        ax: plt.Axes | None = None,
    ) -> None:
        fill = ax is not None

        if not fill:
            fig, ax = plt.subplots(ncols=1, figsize=(6, 4), tight_layout=True)

        lb, ub = bounds
        x = 1e-9*np.linspace(lb, ub, 1000)
        y = self.responce(x)
        ax.plot(
            1e+9*x, y,
            color='black',
        )
        text = f'Detector: {self.config.name}'
        match info:
            case 'title':
                ax.set_title(text)
            case 'text':
                ax.text(
                    0.05, 0.95,
                    text,
                    transform=ax.transAxes,
                    ha='left', va='top',
                )
            case 'none':
                pass

        ax.set_xlim(bounds)
        ax.set_ylim([0, 1.25])

        ax.set_xlabel(r'$\lambda, нм$')
        ax.set_ylabel(r'Спектральный отклик фотодиода, А/Вт')

        ax.grid(color='grey', linestyle=':')

        if save:
            filepath = os.path.join('.', 'report', 'img', 'Detector-response.png')
            plt.savefig(filepath)

        if not fill:
            plt.show()

    def __str__(self) -> str:
        cls = self.__class__
        name = self.config.name

        return f'{cls.__name__}: {name}'
