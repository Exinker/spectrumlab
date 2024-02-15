import os
from dataclasses import dataclass

import pandas as pd

from spectrumlab.alias import Frame, Kelvin, Symbol


DATABASE_DIRECTORY = os.path.join(os.path.dirname(__file__), 'database')
DATABASE_VERSION = '0.01'


@dataclass
class Element:
    symbol: Symbol
    atomic_number: int
    name: str
    atomic_weight: float  # in da (dalton)
    density: float  # in kg/m3
    melting_temperature: Kelvin
    boiling_temperature: Kelvin
    link: str


class PeriodicTable:

    def __init__(self):
        self._version = DATABASE_VERSION
        self._filepath = os.path.join(DATABASE_DIRECTORY, f'database v{self.version}.csv')
        self._database = pd.read_csv(
            self._filepath,
            sep=', ',
            header=0,
        ).set_index('atomic_number', drop=False)

    @property
    def version(self) -> str:
        return self._version

    @property
    def database(self) -> Frame:
        return self._database

    # --------            private            --------
    def __getitem__(self, index: int | str) -> Element:

        if isinstance(index, int):
            db = self._database.set_index('atomic_number', drop=False)
            return Element(**db.loc[index])
        if isinstance(index, str):
            db = self._database.set_index('symbol', drop=False)
            return Element(**db.loc[index])

        raise TypeError()  # TODO: add custom exception

    def __repr__(self) -> str:
        cls = self.__class__

        return f'{cls}({self._filepath})'

    def __str__(self) -> str:
        return str(self._database)
