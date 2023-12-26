from typing import Callable

from spectrumlab.alias import Nano, Number


def default_handler(wavelength: Nano) -> Number:
    """Default wavelength handler (do nothing)."""

    return wavelength


class Line:

    def __init__(self, id: int, symbol: str, wavelength: Nano, *args, database_intensity: float = 0, database_ionization_degree: int = 1, handler: Callable[[Nano], Number] | None = None, **kwargs):

        self.id = id
        self.symbol = symbol
        self.wavelength = wavelength
        self.nickname = f'{symbol} {wavelength}'
        self.database_intensity = database_intensity
        self.database_ionization_degree = database_ionization_degree

        self._handler = handler or default_handler

    @property
    def position(self) -> Number:
        return self._handler(self.wavelength)

    @property
    def intensity(self) -> Number:
        return self.database_intensity

    # --------            others            --------
    def __repr__(self) -> str:
        cls = self.__class__

        content = '; '.join([
            f'{self.nickname}',
            f'id: {self.id}',
        ])
        return f'{cls.__name__}({content})'


if __name__ == '__main__':
    line = Line(
        id=0,
        symbol='Cu',
        wavelength=324.75,
    )
    print(line)
