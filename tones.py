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

class Samples(list):
    """
    Extension of list class with methods useful for manipulating audio samples
    """

    def _pack_sample(self, sample):
        ret = int(sample * MAX_SAMPLE_VALUE)
        maxp = int(MAX_SAMPLE_VALUE)

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
    Represents a fixed monophonic tone
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

        :param int wavetype: waveform type
        :param float frequency: tone frequency
        :param int rate: sample rate for generating samples
        :param float amplitude: Tone amplitude, where 1.0 is the max. sample \
            value and 0.0 is total silence
        """

        try:
            self.tablefunc = self.table_generators[wavetype]
        except KeyError:
            raise ValueError("Invalid wave type: %s" % wavetype)

        self._amp = amplitude
        self._rate = rate
        self._period = int(rate / frequency)
        self._table = self.tablefunc(frequency, rate, amplitude)

    def slide(self, num, start, end):
        time_step = 0.01
        sample_step = int(time_step * self._rate) # change frequency every 10ms
        num_steps = int(num / sample_step)
        freq_delta = end - start
        freq_step = freq_delta / num_steps

        i = 0
        last_size = 0
        freq = float(start)
        ret = Samples()

        while freq <= end:
            table = self.tablefunc(freq, self._rate, self._amp)
            period = len(table)
            if last_size > 0:
                i = int((float(i) / (last_size - 1)) * (period - 1))

            for _ in range(sample_step):
                ret.append(table[i % period])
                i += 1

            i %= period
            last_size = period
            freq += freq_step

        return ret

    def samples(self, num, attack=0.05, decay=0.05):
        """
        Generate tone for a specific number of samples

        :param int num: number of samples to generate
        :param float attack: tone attack in seconds
        :param float decay: tone decay in seconds
        :return: samples in the range of -1.0 to 1.0
        :rtype: Samples
        """

        ret = Samples()

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

        :param int wavetype: initial wavetype setting for this track
        """

        self._samples = Samples()
        self._wavetype = wavetype
        self._weighting = None

    def append_samples(self, samples):
        """
        Append samples to this track. Samples should be in the range -1.0 to 1.0

        :param [float] samples: samples to append
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

        :param int sample_rate: sampling rate in Hz
        :param float amplitude: master amplitude in the range 0.0 to 1.0
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

    def _silence(self, samples):
        return Samples([0.0] * samples)

    def create_track(self, trackname, wavetype=SINE_WAVE):
        """
        Creates a Tone track

        :param trackname: unique identifier for track. Can be any hashable type.
        :param int wavetype: initial wavetype setting for track
        """

        self._tracks[trackname] = Track(wavetype)

    def add_samples(self, trackname, samples):
        """
        Adds samples to a track

        :param trackname: track identifier, track to add samples to
        :param [float] samples: samples to add
        """

        track = self._get_track(trackname)
        track.append_samples(samples)

    def add_tone(self, trackname, frequency=440.0, duration=1.0, attack=0.1,
            decay=0.1, amplitude=1.0):
        """
        Create a tone and add the samples to a pecified track

        :param trackname: track identifier, track to add tone to
        :param float frequency: tone frequency
        :param float duration: tone duration in seconds
        :param float attack: tone attack in seconds
        :param float decay: tone decay in seconds
        :param float amplitude: Tone amplitude, where 1.0 is the max. sample \
            value and 0.0 is total silence
        """

        track = self._get_track(trackname)
        tone = Tone(frequency, self._rate, amplitude, track._wavetype)
        numsamples = int(duration * self._rate)
        track.append_samples(tone.samples(numsamples, attack, decay))

    def add_silence(self, trackname, duration=1.0):
        """
        Adds silence to a track

        :param trackname: track identifier, track to add silence to
        :param float duration: silence duration in seconds
        """

        track = self._get_track(trackname)
        track.append_samples(self._silence(duration * self._rate))

    def set_wavetype(self, trackname, wavetype):
        """
        Sets the waveform type for a track

        :param trackname: track identifier, track to set wavetype for
        :param int wavetype: waveform type
        """

        track = self._get_track(trackname)
        track._wavetype = wavetype

        track = self._get_track(trackname)
        track._weighting = weighting

    def mix(self):
        """
        Mixes all tracks into a single stream of samples

        :return: mixed samples
        :rtype: Samples
        """

        if len(self._tracks) == 0:
            return []

        tracks = self._tracks.values()
        tracks.sort(key=lambda x: len(x._samples), reverse=True)

        default_div = len(tracks)
        weight = 1.0 / default_div

        ret = self._silence(len(tracks[0]._samples))

        for track in tracks:
            for i in range(len(track._samples)):
                ret[i] += (track._samples[i] * weight) * self._amp

        return ret

    def write_wav(self, filename):
        """
        Mixes all tracks into a single stream of samples and writes to a
        .wav audio file

        :param str filename: name of file to write
        """

        samples = self.mix()

        f = wave.open(filename, 'w')
        f.setparams((NUM_CHANNELS, DATA_SIZE, self._rate, len(samples),
            "NONE", "Uncompressed"))

        f.writeframesraw(samples.serialize())
        f.close()

def main():
    m = Mixer(44100, 0.5)
    m.create_track(0, SINE_WAVE)
    #m.create_track(1, SINE_WAVE)
    #m.create_track(2, SINE_WAVE)

    #m.add_tone(0, frequency=440.0, duration=1.0, attack=0.1, decay=0.5)
    #m.add_tone(1, frequency=450.0, duration=1.0, attack=0.1, decay=0.5)
    #m.add_tone(2, frequency=460.0, duration=1.0, attack=0.1, decay=0.5)
    t = Tone(440.0, 44100, 0.5, SINE_WAVE)
    m.add_samples(0, t.slide(44100, 440.0, 1000.0))
    m.write_wav('super.wav')

if __name__ == "__main__":
    main()
