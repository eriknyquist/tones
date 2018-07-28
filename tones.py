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

def _translate(value, inmin, inmax, outmin, outmax):
    scaled = float(value - inmin) / float(inmax - inmin)
    return outmin + (scaled * (outmax - outmin))

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

    _pitch_time_step = 0.001

    _table_generators = {
        SINE_WAVE: _sine_wave_table,
        SQUARE_WAVE: _square_wave_table,
        TRIANGLE_WAVE: _triangle_wave_table,
        SAWTOOTH_WAVE: _sawtooth_wave_table
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
            points.append(_translate(point, 0.0, 1.0, -half, half))
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
        period = int(self._rate / frequency)

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
            i = self._phase_to_index(phase, len(table))

            for _ in range(num):
                samples.append(table[i % period])
                i += 1

            phase = self._index_to_phase(i % period, len(table))
        else:
            samples, phase = self._variable_pitch_tone(points, phase)

        if attack and attack > 0.0:
            step = 1.0 / (self._rate * attack)
            _fade_up(samples, 0, len(samples), 1, step)

        if decay and decay > 0.0:
            step = 1.0 / (self._rate * decay)
            _fade_up(samples, len(samples) - 1, -1, -1, step)

        return samples, phase, vphase

    @staticmethod
    def _index_to_phase(index, size):
        return (float(index) / (size - 1)) * 359.0

    @staticmethod
    def _phase_to_index(phase, size):
        return int((float(phase) / 359.0) * (size - 1))

class Track(object):
    """
    Represents a single track in a Mixer
    """

    def __init__(self, wavetype=SINE_WAVE, attack=None, decay=None,
            vibrato_frequency=None, vibrato_variance=15.0):
        """
        Initializes a Track

        :param int wavetype: initial wavetype setting for this track
        :param float attack: initial tone attack for this track, to applied to \
            each set of samples generated by 'append_samples'
        :param float decay: initial tone decay for this track, to applied to \
            each set of samples generated by 'append_samples'
        :param float vibrato_frequency: initial vibrato frequency for this track
        :param float vibrato_variance: initial vibrato variance for this track
        """

        self._attack = attack
        self._decay = decay
        self._vibrato_frequency = vibrato_frequency
        self._vibrato_variance = vibrato_variance
        self._phase = 0.0
        self._vphase = 0.0
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

    _notes = {
        "c": 261.626,
        "c#": 277.183,
        "db": 277.183,
        "d": 293.665,
        "d#": 311.127,
        "eb": 311.127,
        "e": 329.628,
        "e#": 349.228,
        "f": 349.228,
        "f#": 369.994,
        "gb": 369.994,
        "g": 391.995,
        "g#": 415.305,
        "ab": 415.305,
        "a": 440.0,
        "a#": 466.164,
        "bb": 466.164,
        "b": 493.883
    }


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

    def _get_note(self, note, octave):
        try:
            freq = self._notes[note.lower()]
        except KeyError:
            raise ValueError("invalid note: %s" % note)

        if octave < 4:
            freq /= math.pow(2, (4 - octave))
        elif octave > 4:
            freq *= math.pow(2, (octave - 4))

        return freq

    def _silence(self, samples):
        return Samples([0.0] * samples)

    def create_track(self, trackname, *args, **kwargs):
        """
        Creates a Tone track

        :param trackname: unique identifier for track. Can be any hashable type.
        :param args: arguments for Track constructor
        :param kwargs: keyword arguments for Track constructor
        """

        self._tracks[trackname] = Track(*args, **kwargs)

    def set_attack(self, trackname, attack):
        """
        Set the tone attack for a track. This attack will be applied to
        all tones added to this track.

        :param trackname: track identifier
        :param float attack: attack time in seconds
        """

        track = self._get_track(trackname)
        track._attack = attack

    def get_attack(self, trackname):
        """
        Get the tone attack for a track

        :param trackname: track identifier
        :return: tone attack
        :rtype: float
        """

        track = self._get_track(trackname)
        return track._attack

    def set_decay(self, trackname, decay):
        """
        Set tone decay for a track. This decay will be applied to all tones
        added to this track

        :param trackname: track identifier
        :param float decay: decay time in seconds
        """

        track = self._get_track(trackname)
        track._decay = decay

    def get_decay(self, trackname):
        """
        Get the tone decay for a track

        :param trackname: track identifier
        :return: tone decay
        :rtype: float
        """

        track = self._get_track(trackname)
        return track._decay

    def set_vibrato_frequency(self, trackname, frequency):
        """
        Set vibrato frequency for a track. This vibrato frequency will be
        applied to all tones added to this track

        :param trackname: track identifier
        :param float frequency: vibrato frequency in Hz
        """

        track = self._get_track(trackname)
        track._vibrato_frequency = frequency

    def get_vibrato_frequency(self, trackname):
        """
        Get the vibrato frequency for a track

        :param trackname: track identifier
        :return: vibrato frequency in Hz
        :rtype: float
        """

        track = self._get_track(trackname)
        return track._vibrato_frequency

    def set_vibrato_variance(self, trackname, variance):
        """
        Set vibrato variance for a track. The variance represents the full range
        that the highest and lowest points of the vibrato will reach, in Hz; for
        example, a tone at 440Hz with a vibrato variance of 20hz would
        oscillate between 450Hz and 430Hz. This vibrato variance will be
        applied to all tones added to this track

        :param trackname: track identifier
        :param float variance: vibrato variance in Hz
        """

        track = self._get_track(trackname)
        track._vibrato_variance = variance

    def get_vibrato_variance(self, trackname):
        """
        Get the vibrato variance for a track

        :param trackname: track identifier
        :return: vibrato variance in Hz
        :rtype: float
        """

        track = self._get_track(trackname)
        return track._vibrato_variance

    def add_samples(self, trackname, samples):
        """
        Adds samples to a track

        :param trackname: track identifier, track to add samples to
        :param [float] samples: samples to add
        """

        track = self._get_track(trackname)
        track.append_samples(samples)

    def add_tone(self, trackname, frequency=440.0, duration=1.0,
            endfrequency=None, attack=None, decay=None, amplitude=1.0,
            vibrato_frequency=None, vibrato_variance=None):
        """
        Create a tone and add the samples to a track

        :param trackname: track identifier, track to add tone to
        :param float frequency: tone frequency
        :param float duration: tone duration in seconds
        :param float attack: tone attack in seconds. Overrides the track's \
            attack setting
        :param float decay: tone decay in seconds. Overrides the track's \
            decay setting
        :param float vibrato_frequency: tone vibrato frequency in Hz. Overrides\
            the track's vibrato frequency setting
        :param float vibrato_variance: tone vibrato variance in Hz. Overrides\
            the track's vibrato variance setting
        :param float amplitude: Tone amplitude, where 1.0 is the max. sample \
            value and 0.0 is total silence
        """

        track = self._get_track(trackname)
        tone = Tone(self._rate, amplitude, track._wavetype)
        numsamples = int(duration * self._rate)

        if not attack:
            attack = track._attack
        if not decay:
            decay = track._decay
        if not vibrato_frequency:
            vibrato_frequency = track._vibrato_frequency
        if not vibrato_variance:
            vibrato_variance = track._vibrato_variance

        samples, track._phase, track._vphase = tone.samples(numsamples,
                frequency, endfrequency, attack, decay, track._phase,
                track._vphase, vibrato_frequency, vibrato_variance)

        track.append_samples(samples)

    def add_note(self, trackname, note="a", octave=4, duration=1.0,
            endnote=None, endoctave=None, attack=None, decay=None,
            amplitude=1.0, vibrato_frequency=None, vibrato_variance=None):
        """
        Same as 'add_tone', except the pitch can be specified as a standard
        musical note, e.g. "c#"

        :param trackname: track identifier, track to add tone to
        :param str note: musical note. Must be a single character from a-g \
            (non case-sensitive), followed by an optional sharp ('#') or flat \
            ('b') character
        :param str endnote: If not None, the tone frequency will change \
            between 'note' and 'endnote' in increments of 1ms over \
            all samples
        :param int octave: note octave from 0-8
        :param int endoctave: octave for the note specified by endnote
        :param float duration: tone duration in seconds
        :param float attack: tone attack in seconds. Overrides the track's \
            attack setting
        :param float decay: tone decay in seconds. Overrides the track's \
            decay setting
        :param float vibrato_frequency: tone vibrato frequency in Hz. Overrides\
            the track's vibrato frequency setting
        :param float vibrato_variance: tone vibrato variance in Hz. Overrides\
            the track's vibrato variance setting
        :param float amplitude: Tone amplitude, where 1.0 is the max. sample \
            value and 0.0 is total silence
        """

        endfreq = None

        freq = self._get_note(note, octave)
        if endnote:
            if not endoctave:
                endoctave = octave

            endfreq = self._get_note(endnote, endoctave)

        self.add_tone(trackname, freq, duration, endfreq, attack, decay,
            amplitude, vibrato_frequency, vibrato_variance)

    def add_tones(self, tonelist):
        """
        Create multiple tones and add the samples for each tone in order to a
        track

        :param tonelist: list of tuples, where each tuple contains arguments \
            for a single Mixer.add_tone invocation
        """

        for arglist in tonelist:
            self.add_tone(*arglist)

    def add_notes(self, trackname, notelist):
        """
        Create multiple notes and add the samples for each tone in order to a
        track

        :param trackname: track identifier
        :param notelist: list of tuples, where each tuple contains arguments \
            for a single Mixer.add_note invocation
        """

        for arglist in notelist:
            self.add_note(trackname, *arglist)

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

    def get_wavetype(self, trackname):
        """
        Get the waveform type for a track

        :param trackname: track identifier
        :return: track waveform type
        :rtype: int
        """

        track = self._get_track(trackname)
        return track._wavetype

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

def _main():
    m = Mixer(44100, 0.5)
    m.create_track(0, SINE_WAVE, vibrato_frequency=7.0)
    m.create_track(1, SINE_WAVE)

    m.add_note(0, note='c', octave=5, duration=1.0)
    m.add_note(0, note='c', octave=5, duration=0.2, endnote='d')
    m.add_note(0, note='d', octave=5, duration=1.0)
    m.add_note(0, note='d', octave=5, duration=0.2, endnote='c')
    m.add_note(0, note='c', octave=5, duration=1.0)
    m.add_note(0, note='c', octave=5, duration=1.0)
    m.add_note(0, note='c', octave=5, duration=1.0, endnote='f', endoctave=5, vibrato_variance=30, attack=None, decay=1.0)

    m.add_notes(1, [
        ('d', 5, 1.0),
        ('d', 5, 0.2, 'f'),
        ('f', 5, 1.0),
        ('f', 5, 0.2, 'd'),
        ('d', 5, 1.0, None, None, None, 0.2, 1.0, 7, 30)
    ])

    m.write_wav('super.wav')

if __name__ == "__main__":
    _main()
