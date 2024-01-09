from spectrumlab.emulation.apparatus import VoigtApparatusShape
from spectrumlab.emulation.detector.linear_array_detector import Detector

IS_NOISED = False

N_NUMBERS = 20
N_FRAMES = 1
N_ITERS = 51

DETECTOR = Detector.BLPP2000
SHAPE = VoigtApparatusShape(
    width=28,
    asymmetry=+0.1,
    ratio=0.1,
)

EXPOSURE = 100
POSITION = N_NUMBERS//2
INTENSITY = 1
