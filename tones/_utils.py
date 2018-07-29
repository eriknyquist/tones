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

def _sine_sample(amp, freq, period, rate, i):
    return float(amp) * math.sin(2.0 * math.pi * float(freq)
        * (float(i % period) / float(rate)))
