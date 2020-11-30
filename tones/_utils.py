import math

import tones

def _fade_up(data, start, end, istep=1, astep=0.005):
    amp = 0.0
    for i in range(start, end, istep):
        if amp >= 1.0:
            break

        data[i] *= amp
        amp += astep

def _translate(value, inmin, inmax, outmin, outmax):
    scaled = float(value - inmin) / float(inmax - inmin)
    return outmin + (scaled * (outmax - outmin))

def _sine_sample(amp, freq, rate, i):
    """
    Generates a single audio sample taken at the given sampling rate on
    a sine wave oscillating at the given frequency at the given amplitude.

    :param float amp The amplitude of the sine wave to sample
    :param float freq The frequency of the sine wave to sample
    :param int rate The sampling rate
    :param int i The index of the sample to pull

    :return float The audio sample as described above
    """
    return float(amp) * math.sin(2.0 * math.pi * float(freq)
        * (float(i) / float(rate)))

def _triangle_sample(amp, freq, rate, i):
    """
    Generates a single audio sample taken at the given sampling rate on
    a triangle wave oscillating at the given frequency at the given amplitude.

    :param float amp The amplitude of the sine wave to sample
    :param float freq The frequency of the sine wave to sample
    :param int rate The sampling rate
    :param int i The index of the sample to pull

    :return float The audio sample as described above
    """
    return float(amp) * math.asin(math.sin(2.0 * math.pi * float(freq)
        * (float(i) / float(rate))))

def _sawtooth_sample(amp, freq, rate, i):
    """
    Generates a single audio sample taken at the given sampling rate on
    a sawtooth wave oscillating at the given frequency at the given amplitude.

    :param float amp The amplitude of the sine wave to sample
    :param float freq The frequency of the sine wave to sample
    :param int rate The sampling rate
    :param int i The index of the sample to pull

    :return float The audio sample as described above
    """
    return float(amp) * math.atan(math.tan(2.0 * math.pi * float(freq)
        * (float(i) / float(rate))))
