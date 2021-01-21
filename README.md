# real_time_speech_denoiser
Real time application to show speech denoising in action.

# Implementation details

- I decided to use the python package sounddevice to capture and play audio.
  - To see the documentation of this package, see  https://python-sounddevice.readthedocs.io/en/0.4.1/
  - The reason I chose this library, is that it can store the audio it records,
      and play audio, from numpy arrays directly, in addition to being able to
      work with bytes (and maybe even streams).
  - Another reason I chose this library, is that it is cross-platform.


# TODO
- [] Change class names from Receiver to Reader and Listener to Writer
- [] Move code to separate files outside of POC
- [] Create FileReader (FilePlayer in class diagram)
- [] Create FileWriter (SaveToFile in class diagram)
- [] Create Processor Listener (Gets a Processor and a Listener, and puts everything it gets through the processor and sends it’s output to the listener)
- [] Create Splitter Listener (can be created by a Splitter Processor)
- [] Create Player Listener
- [] Create sequence diagram
- [] Tidy up code
- [] Add docstring to everything in the code
- [] Add details and classes to class diagram
- [] Update README with diagrams and startup instructions
- [] Create requirements.txt with sections for visualizer / sounddevice / etc
- [] Simplify gitignore
- [] Make code robust (make sure nothing is hard coded, everything is according to specs, try on different systems, etc)
- [] Change abstract classes to really be abstract (or interfaces if possible)
- [] Make into package (add setyp.py, make imports relative, add tests, etc). Read https://blog.ionelmc.ro/2014/05/25/python-packaging/ and https://github.com/pypa/sampleproject to see what needs to be done.
- [] Create Noise Reduction Processor in a different project
