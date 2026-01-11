from enum import Enum
from pathlib import Path
from typing import NewType, TypeAlias, TypeVar

import pandas as pd
from numpy.typing import NDArray  # noqa: I100

from spectrumlab.picture.colors import COLOR_DATASET, Color


# --------        paths        --------
DirPath: TypeAlias = str | Path
FilePath: TypeAlias = str | Path


# --------        structures        --------
Array: TypeAlias = NDArray

Index: TypeAlias = pd.Index | pd.MultiIndex
Series: TypeAlias = pd.Series
Frame: TypeAlias = pd.DataFrame


# --------        datasets        --------
class Dataset(Enum):

    train = 'train'
    valid = 'valid'
    test = 'test'

    @property
    def color(self) -> Color:
        return COLOR_DATASET[self.name]


# --------        temperature units        --------
Kelvin = NewType('Kelvin', float)
Celsius = NewType('Celsius', float)


# --------        time units        --------
Second = NewType('Second', float)
MilliSecond = NewType('MilliSecond', float)
MicroSecond = NewType('MicroSecond', int)

Hz = NewType('Hz', float)


# --------        spacial units        --------
Inch = NewType('Inch', float)

Meter = NewType('Meter', float)
CentiMeter = NewType('CentiMeter', float)
MilliMeter = NewType('MilliMeter', float)
MicroMeter = NewType('MicroMeter', float)
NanoMeter = NewType('NanoMeter', float)
PicoMeter = NewType('Pico', float)

Number = NewType('Number', float)

T = TypeVar('T', Number, MicroMeter, NanoMeter, PicoMeter)


# --------        value units        --------
Digit = NewType('Digit', float)
Electron = NewType('Electron', float)
Percent = NewType('Percent', float)

Absorbance = NewType('Absorbance', float)

R = TypeVar('R', Digit, Electron, Percent, Absorbance)
C = NewType('C', float)


# --------        other units        --------
Intercept = NewType('Intercept', float)
Slope = NewType('Slope', float)

Symbol = NewType('Symbol', str)
