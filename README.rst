Tones
=====

A pure-python module for generating simple tones as audio samples, which can
optionally be written directly to a .wav audio file. Supports pitch-bending,
vibrato, polyphony, several waveform types (sine, square, triangle,
sawtooth), and several other waveform-shaping options.

Installation
============

Install from the PyPi repository:

::

    pip install tones

Example
=======

.. code:: python

   from tones import SINE_WAVE, SAWTOOTH_WAVE
   from tones.mixer import Mixer

   # Create mixer, set sample rate and amplitude
   mixer = Mixer(44100, 0.5)

   # Create two monophonic tracks that will play simultaneously, and set
   # initial values for note attack, decay and vibrato frequency (these can
   # be changed again at any time, see documentation for tones.Mixer
   mixer.create_track(0, SAWTOOTH_WAVE, vibrato_frequency=7.0, vibrato_variance=30.0, attack=0.01, decay=0.1)
   mixer.create_track(1, SINE_WAVE, attack=0.01, decay=0.1)

   # Add a 1-second tone on track 0, slide pitch from c# to f#)
   mixer.add_note(0, note='c#', octave=5, duration=1.0, endnote='f#')

   # Add a 1-second tone on track 1, slide pitch from f# to g#)
   mixer.add_note(1, note='f#', octave=5, duration=1.0, endnote='g#')

   # Mix all tracks into a single list of samples and write to .wav file
   mixer.write_wav('tones.wav')
    
   # Mix all tracks into a single list of samples scaled from 0.0 to 1.0, and
   # return the sample list
   samples = mixer.mix()

Documentation
=============

Full API documention is here: `<https://tones.readthedocs.io>`_
