import struct

import tones
import tones._utils as utils

def _sine_wave_table(freq, rate, amp):
    period = int(rate / freq)
    return [utils._sine_sample(amp, freq, period, rate, i) for i in range(period)]

def _square_wave_table(freq, rate, amp):
    ret = []
    for s in _sine_wave_table(freq, rate, amp):
        ret.append(amp if s > 0 else -amp)

    return ret

def _triangle_wave_table(freq, rate, amp):
    period = int(rate / freq)
    slope = 2.0 / (period / 2.0)
    val = 0.0
    step = slope

    ret = []
    for _ in range(period):
        if val >= 1.0:
            step = -slope
        elif val <= -1.0:
            step = slope

        ret.append(amp * val)
        val += step

    return ret

def _sawtooth_wave_table(freq, rate, amp):
    period = int(rate / freq)
    slope = 2.0 / period
    val = 0.0

    ret = []
    for _ in range(period):
        if val >= 1.0:
            val = -1.0

        ret.append(val * amp)
        val += slope

    return ret

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

    _table_generators = {
        tones.SINE_WAVE: _sine_wave_table,
        tones.SQUARE_WAVE: _square_wave_table,
        tones.TRIANGLE_WAVE: _triangle_wave_table,
        tones.SAWTOOTH_WAVE: _sawtooth_wave_table
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
            self.tablefunc = self._table_generators[wavetype]
        except KeyError:
            raise ValueError("Invalid wave type: %s" % wavetype)

        self._amp = amplitude
        self._rate = rate

    def _variable_pitch_tone(self, points, phase):
        sample_step = int(self._pitch_time_step * self._rate)

        i = 0
        ret = Samples()

        for freq in points:
            table = self.tablefunc(freq, self._rate, self._amp)
            period = len(table)
            i = self._phase_to_index(phase, period)

            for _ in range(sample_step):
                ret.append(table[i % period])
                i += 1

            phase = self._index_to_phase(i % period, period)

        return ret, phase

    def _vibrato_pitch_change(self, numsamples, freq, variance, phase):
        stepsamples = self._pitch_time_step * self._rate
        numsteps = float(numsamples) / stepsamples
        points = []
        half = variance / 2.0

        table = _sine_wave_table(freq, int(1.0 / self._pitch_time_step), 1.0)
        period = len(table)
        i = self._phase_to_index(phase, len(table))

        for _ in range(int(numsteps)):
            point = table[i % period]
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
            table = self.tablefunc(frequency, self._rate, self._amp)
            period = len(table)
            i = self._phase_to_index(phase, period)

            for _ in range(num):
                samples.append(table[i % period])
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
