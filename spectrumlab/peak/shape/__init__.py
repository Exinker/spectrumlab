
from typing import TypeAlias

from .peak_shape import VoightPeakShape, EffectedVoightPeakShape


PeakShape: TypeAlias = VoightPeakShape | EffectedVoightPeakShape
