import wave
import math
import struct

SINE_WAVE = 0
SQUARE_WAVE = 1
TRIANGLE_WAVE = 2
SAWTOOTH_WAVE = 3

DATA_SIZE = 2
NUM_CHANNELS = 1
MAX_SAMPLE_VALUE = float(int((2 ** (DATA_SIZE * 8)) / 2) - 1)

def _pack_sample(sample):
    ret = int(sample * MAX_SAMPLE_VALUE)
    maxp = int(MAX_SAMPLE_VALUE)

    if ret < -maxp:
        ret = -maxp
    elif ret > maxp:
        ret = maxp

    return struct.pack('h', ret)

def _serialize_samples(samples):
    return bytes(b'').join([bytes(_pack_sample(s)) for s in samples])

def _write_wav_file(samples, filename):
    f = wave.open(filename, 'w')
    f.setparams((NUM_CHANNELS, DATA_SIZE, 44100, len(samples),
        "NONE", "Uncompressed"))
    f.writeframesraw(_serialize_samples(samples))
    f.close()

def _fade_up(data, start, end, istep=1, astep=0.005):
    amp = 0.0
    for i in range(start, end, istep):
        if amp >= 1.0:
            break

        data[i] *= amp
        amp += astep

def _sine_sample(amp, freq, period, rate, i):
    return float(amp) * math.sin(2.0 * math.pi * float(freq)
        * (float(i % period) / float(rate)))

def _sine_wave_table(freq, rate, amp):
    period = int(rate / freq)
    return [_sine_sample(amp, freq, period, rate, i) for i in range(period)]

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

class Tone(object):
    """
    Represents a monophonic tone
    """

    table_generators = {
        SINE_WAVE: _sine_wave_table,
        SQUARE_WAVE: _square_wave_table,
        TRIANGLE_WAVE: _triangle_wave_table,
        SAWTOOTH_WAVE: _sawtooth_wave_table
    }

    def __init__(self, frequency, rate, amplitude, wavetype):
        """
        Initializes a Tone

        :param wavetype: waveform type
        :param frequency: tone frequency
        :type frequency: float
        :param rate: sample rate for generating samples
        :type rate: int
        :param amplitude: Tone amplitude, where 1.0 is the max. sample value\
            and 0.0 is total silence
        :type amplitude: float
        """

        try:
            tablefunc = self.table_generators[wavetype]
        except KeyError:
            raise ValueError("Invalid wave type: %s" % wavetype)

        self._rate = rate
        self._period = int(rate / frequency)
        self._table = tablefunc(frequency, rate, amplitude)

    def samples(self, num, attack=0.05, decay=0.05):
        """
        Generate tone for a specific number of samples

        :param num: number of samples to generate
        :type num: int
        :param attack: tone attack in seconds
        :type attack: float
        :param decay: tone decay in seconds
        :type decay: float
        :return: samples in the range of -1.0 to 1.0
        :rtype: [float]
        """

        ret = []

        for i in range(num):
            ret.append(self._table[i % self._period])

        if attack and attack > 0.0:
            step = 1.0 / (self._rate * attack)
            _fade_up(ret, 0, len(ret), 1, step)

        if decay and decay > 0.0:
            step = 1.0 / (self._rate * decay)
            _fade_up(ret, len(ret) - 1, -1, -1, step)

        return ret

class Track(object):
    """
    Represents a single track in a Mixer
    """

    def __init__(self, wavetype=SINE_WAVE):
        """
        Initializes a Track

        :param wavetype: initial wavetype setting for this track
        :type wavetype: int
        """

        self._samples = []
        self._wavetype = wavetype
        self._weighting = None

    def append_samples(self, samples):
        """
        Append samples to this track. Samples should be in the range -1.0 to 1.0

        :param samples: samples to append
        :type samples: [float]
        """

        self._samples.extend(samples)

class Mixer(object):
    """
    Represents multiple tracks that can be summed together into a single
    list of samples
    """

    def __init__(self, sample_rate=44100, amplitude=0.5):
        """
        Initializes a Mixer

        :param sample_rate: sampling rate in Hz
        :type sample_rate: int
        :param amplitude: master amplitude in the range 0.0 to 1.0
        :type amplitude: float
        """

        self._rate = sample_rate
        self._amp = amplitude
        self._tracks = {}

    def _get_track(self, trackname):
        try:
            ret = self._tracks[trackname]
        except KeyError:
            raise ValueError("No such track '%s'" % trackname)

        return ret

    def create_track(self, trackname, wavetype=SINE_WAVE):
        self._tracks[trackname] = Track(wavetype)

    def add_tone(self, trackname, frequency=440.0, duration=1.0, attack=0.1,
            decay=0.1, amplitude=1.0):
        """
        """

        track = self._get_track(trackname)
        tone = Tone(frequency, self._rate, amplitude, track._wavetype)
        numsamples = int(duration * self._rate)
        samples = tone.samples(numsamples, attack, decay)
        track.append_samples(samples)

    def set_wavetype(self, trackname, wavetype):
        track = self._get_track(trackname)
        track._wavetype = wavetype

    def set_weighting(self, trackname, weighting):
        track = self._get_track(trackname)
        track._weighting = weighting

    def mix(self, use_weightings=False):
        if len(self._tracks) == 0:
            return []

        tracks = self._tracks.values()
        tracks.sort(key=lambda x: len(x._samples), reverse=True)

        default_div = len(tracks)

        def _with_weights(track):
            return track._weighting

        def _without_weights(track):
            return 1.0 / default_div

        _getweight = _with_weights if use_weightings else _without_weights
        ret = [0.0] * len(tracks[0]._samples)

        for track in tracks:
            for i in range(len(track._samples)):
                ret[i] += (track._samples[i] * _getweight(track)) * self._amp

        return ret

def main():
    m = Mixer(44100, 0.5)
    m.create_track(0, SINE_WAVE)
    m.create_track(1, SINE_WAVE)
    m.create_track(2, SINE_WAVE)

    m.add_tone(0, frequency=440.0, duration=1.0, attack=0.1, decay=0.5)
    m.add_tone(1, frequency=450.0, duration=1.0, attack=0.1, decay=0.5)
    m.add_tone(2, frequency=460.0, duration=1.0, attack=0.1, decay=0.5)
    _write_wav_file(m.mix(), 'super.wav')

if __name__ == "__main__":
    main()
