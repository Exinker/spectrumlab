
from typing import TypeAlias

from .peak_profile import VoightPeakProfile, EffectedVoightPeakProfile


PeakProfile: TypeAlias = VoightPeakProfile | EffectedVoightPeakProfile
