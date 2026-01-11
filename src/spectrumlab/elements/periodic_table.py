import os

import pandas as pd

from spectrumlab.elements import Element
from spectrumlab.types import DirPath, Frame


DATABASE_DIRECTORY = os.path.join(os.path.dirname(__file__), 'database')
DATABASE_VERSION = '0.01'


class PeriodicTable:

    def __init__(self, version: str = DATABASE_VERSION, filedir: DirPath = DATABASE_DIRECTORY):

        self._version = version
        self._filedir = filedir
        self._database = pd.read_csv(
            self.filepath,
            sep=';',
            header=0,
        ).set_index('atomic_number', drop=False)

    @property
    def version(self) -> str:
        return self._version

    @property
    def filedir(self) -> str:
        return self._filedir

    @property
    def filepath(self) -> str:
        return os.path.join(self.filedir, f'database v{self.version}.csv')

    @property
    def database(self) -> Frame:
        return self._database

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

        return f'{cls}(v{self._version})'

    def __str__(self) -> str:
        return str(self._database)
