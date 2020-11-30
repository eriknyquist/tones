__author__ = "Erik Nyquist"
__license__ = "Apache 2.0"
__version__ = "1.2.0"
__maintainer__ = "Erik Nyquist"
__email__ = "eknyquist@gmail.com"


SINE_WAVE = 0
SQUARE_WAVE = 1
TRIANGLE_WAVE = 2
SAWTOOTH_WAVE = 3

DATA_SIZE = 2
NUM_CHANNELS = 1
MAX_SAMPLE_VALUE = float(int((2 ** (DATA_SIZE * 8)) / 2) - 1)
