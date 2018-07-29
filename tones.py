from tones import SINE_WAVE
from tones.mixer import Mixer

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
