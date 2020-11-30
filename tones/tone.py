import struct
from typing import List

import tones
import tones._utils as utils

def _sine_wave_samples(freq, rate, amp, num):
    """
    Generates a set of audio samples taken at the given sampling rate 
    representing a sine wave oscillating at the given frequency with 
    the given amplitude lasting for the given duration.

    :param float freq The frequency of oscillation of the sine wave
    :param int rate The sampling rate
    :param float amp The amplitude of the sine wave
    :param float num The number of samples to generate.

    :return List[float] The audio samples representing the signal as 
                        described above.
    """
    return [utils._sine_sample(amp, freq, rate, i) for i in range(num)]

def _square_wave_samples(freq, rate, amp, num):
    """
    Generates a set of audio samples taken at the given sampling rate 
    representing a square wave oscillating at the given frequency with 
    the given amplitude lasting for the given duration.

    :param float freq The frequency of oscillation of the square wave
    :param int rate The sampling rate
    :param float amp The amplitude of the square wave
    :param float num The number of samples to generate.

    :return List[float] The audio samples representing the signal as 
                        described above.
    """
    ret = []
    for s in _sine_wave_samples(freq, rate, amp, num):
        ret.append(amp if s > 0 else -amp)

    return ret

def _triangle_wave_samples(freq, rate, amp, num):
    """
    Generates a set of audio samples taken at the given sampling rate 
    representing a triangle wave oscillating at the given frequency with 
    the given amplitude lasting for the given duration.

    :param float freq The frequency of oscillation of the triangle wave
    :param int rate The sampling rate
    :param float amp The amplitude of the triangle wave
    :param float num The number of samples to generate.

    :return List[float] The audio samples representing the signal as 
                        described above.
    """
    return [utils._triangle_sample(amp, freq, rate, i) for i in range(num)]

def _sawtooth_wave_samples(freq, rate, amp, num):
    """
    Generates a set of audio samples taken at the given sampling rate 
    representing a sawtooth wave oscillating at the given frequency with 
    the given amplitude lasting for the given duration.

    :param float freq The frequency of oscillation of the sawtooth wave
    :param int rate The sampling rate
    :param float amp The amplitude of the sawtooth wave
    :param float num The number of samples to generate.

    :return List[float] The audio samples representing the signal as 
                        described above.
    """
    return [utils._sawtooth_sample(amp, freq, rate, i) for i in range(num)]

class Samples(list):
    """
    Extension of list class with methods useful for manipulating audio samples
    """

    def _pack_sample(self, sample):
        ret = int(sample * tones.MAX_SAMPLE_VALUE)
        maxp = int(tones.MAX_SAMPLE_VALUE)

        if ret < -maxp:
            ret = -maxp
        elif ret > maxp:
            ret = maxp

        return struct.pack('h', ret)

    def serialize(self):
        """
        Serializes all samples

        :return: serialized samples
        :rtype: bytes
        """

        return bytes(b'').join([bytes(self._pack_sample(s)) for s in self])

class Tone(object):
    """
    Represents a fixed monophonic tone
    """

    _pitch_time_step = 0.001

    _sample_generators = {
        tones.SINE_WAVE: _sine_wave_samples,
        tones.SQUARE_WAVE: _square_wave_samples,
        tones.TRIANGLE_WAVE: _triangle_wave_samples,
        tones.SAWTOOTH_WAVE: _sawtooth_wave_samples
    }

    def __init__(self, rate, amplitude, wavetype):
        """
        Initializes a Tone

        :param int wavetype: waveform type
        :param float frequency: tone frequency
        :param int rate: sample rate for generating samples
        :param float amplitude: Tone amplitude, where 1.0 is the max. sample \
            value and 0.0 is total silence
        """

        try:
            self.samplefunc = self._sample_generators[wavetype]
        except KeyError:
            raise ValueError("Invalid wave type: %s" % wavetype)

        self._amp = amplitude
        self._rate = rate

    def _variable_pitch_tone(self, points, phase):
        sample_step = int(self._pitch_time_step * self._rate)

        i = 0
        ret = Samples()

        for freq in points:
            period = int(self._rate / freq)
            generated_samples = self.samplefunc(freq, self._rate, self._amp, period)
            i = self._phase_to_index(phase, period)

            for _ in range(sample_step):
                ret.append(generated_samples[i % period])
                i += 1

            phase = self._index_to_phase(i % period, period)

        return ret, phase

    def _vibrato_pitch_change(self, numsamples, freq, variance, phase):
        stepsamples = self._pitch_time_step * self._rate
        numsteps = float(numsamples) / stepsamples
        points = []
        half = variance / 2.0

        generated_samples = _sine_wave_samples(freq, int(1.0 / self._pitch_time_step), 1.0, int(numsteps))
        i = self._phase_to_index(phase, len(generated_samples))

        for _ in range(int(numsteps)):
            point = generated_samples[i]
            points.append(utils._translate(point, 0.0, 1.0, -half, half))
            i += 1

        return points, phase

    def _linear_pitch_change(self, numsamples, start, end):
        stepsamples = self._pitch_time_step * self._rate
        numsteps = float(numsamples) / stepsamples
        freqstep = (end - start) / numsteps

        freq = float(start)
        ret = [freq]

        for _ in range(int(numsteps) - 1):
            freq += freqstep
            ret.append(freq)

        return ret

    def samples(self, num, frequency, endfrequency=None, attack=0.05,
            decay=0.05, phase=0.0, vphase=0.0, vibrato_frequency=None,
            vibrato_variance=20.0):
        """
        Generate tone for a specific number of samples

        :param int num: number of samples to generate
        :param float frequency: tone frequency in Hz
        :param float endfrequency: If not None, the tone frequency will change \
            between 'frequency' and 'endfrequency' in increments of 1ms over \
            all samples
        :param float attack: tone attack in seconds
        :param float decay: tone decay in seconds
        :param float phase: starting phase of generated tone in radians
        :param float vphase: starting phase of vibrato in radians
        :param float vibrato_frequency: vibrato frequency in Hz
        :param float vibrato_variance: vibrato variance in Hz
        :return: samples in the range of -1.0 to 1.0, tone phase, vibrato phase
        :rtype: tuple of the form (samples, phase, vibrato_phase)
        """

        points = None

        if not endfrequency is None:
            points = self._linear_pitch_change(num, frequency,
                endfrequency)

        if not vibrato_frequency is None:
            vpoints, vphase = self._vibrato_pitch_change(num, vibrato_frequency,
                vibrato_variance, vphase)

            if points is None:
                points = [frequency + p for p in vpoints]
            else:
                points = [points[i] + vpoints[i] for i in range(len(points))]

        if points is None:
            samples = Samples()
            generated_samples = self.samplefunc(frequency, self._rate, self._amp, num)
            period = len(generated_samples)
            i = 0

            for _ in range(num):
                samples.append(generated_samples[i])
                i += 1

            phase = self._index_to_phase(i % period, period)
        else:
            samples, phase = self._variable_pitch_tone(points, phase)

        if attack and attack > 0.0:
            step = 1.0 / (self._rate * attack)
            utils._fade_up(samples, 0, len(samples), 1, step)

        if decay and decay > 0.0:
            step = 1.0 / (self._rate * decay)
            utils._fade_up(samples, len(samples) - 1, -1, -1, step)
        return samples, phase, vphase

    @staticmethod
    def _index_to_phase(index, size):
        return (float(index) / (size - 1)) * 359.0

    @staticmethod
    def _phase_to_index(phase, size):
        return int((float(phase) / 359.0) * (size - 1))
