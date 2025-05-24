from typing import TypeVar

from spectrumlab.types import (
    Absorbance,
    Digit,
    Electron,
    MicroMeter,
    NanoMeter,
    Number,
    Percent,
    PicoMeter,
)


R = TypeVar('R', Digit, Electron, Percent, Absorbance)
T = TypeVar('T', Number, MicroMeter, NanoMeter, PicoMeter)
