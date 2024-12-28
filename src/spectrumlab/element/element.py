from dataclasses import dataclass

from spectrumlab.types import Kelvin, Symbol


@dataclass
class Element:
    symbol: Symbol
    atomic_number: int
    name: str
    atomic_weight: float  # [da]
    density: float  # [kg/m3]
    melting_temperature: Kelvin
    boiling_temperature: Kelvin
    link: str
