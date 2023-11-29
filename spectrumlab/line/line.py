

class Line:
    def __init__(self, id: int, symbol: str, wavelength: float, database_intensity: float = 0, database_ionization_degree: int = 1, *args, **kwargs):
        self.id = id
        self.symbol = symbol
        self.wavelength = wavelength
        self.nickname = f'{symbol} {wavelength}'
        self.database_intensity = database_intensity
        self.database_ionization_degree = database_ionization_degree

    def __repr__(self) -> str:
        cls = self.__class__

        content = '; '.join([
            f'{self.nickname} ({self.id})',
        ])
        return f'{cls.__name__}({content})'
