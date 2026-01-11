from spectrumlab.types import NanoMeter, Symbol


class Line:

    def __init__(
        self,
        symbol: Symbol,
        wavelength: NanoMeter,
        *args,
        id: int | None = None,
        database_intensity: float = 0,
        database_ionization_degree: int = 1,
        **kwargs,
    ) -> None:
        self._symbol = symbol
        self._wavelength = wavelength

        self._id = id
        self._database_intensity = database_intensity
        self._database_ionization_degree = database_ionization_degree

    @property
    def symbol(self) -> str:
        return self._symbol

    @property
    def wavelength(self) -> NanoMeter:
        return self._wavelength

    @property
    def nickname(self) -> str:
        return f'{self.symbol} {self.wavelength}'

    @property
    def intensity(self) -> float:
        return self._database_intensity

    def __repr__(self) -> str:
        cls = self.__class__

        content = '; '.join([
            f'{self.nickname}',
        ])
        return f'{cls.__name__}({content})'

    def __str__(self) -> str:
        return f'{self.nickname}'


if __name__ == '__main__':
    line = Line(
        symbol='Cu',
        wavelength=324.75,
    )
    print(line)
