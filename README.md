# real_time_speech_denoiser
Real time application to show speech de-noising in action.

# Installing dependencies
In order to use or run this library, some dependencies must be installed on the system. To install
them using the package manager conda, follow the instructions written in requirements.txt and create
a new environment with those packages.

# Running the library
To run an example, be in the home directory of this project, and run
```python -m src.real_time_noise_processing.main.run -f test\config\echo_visualizer.yaml```
and add any other parameters to it.
If you want to run a complex example using a socket tunnel, first start in one terminal:
```python -m src.real_time_noise_processing.main.run -f test\config\socket_visualizer_player.yaml```
and then in a second terminal:
```python -m src.real_time_noise_processing.main.run -f test\config\socket_multiply_socket.yaml```
(this will open a window in the first terminal, which will be unresponsive until the next line runs)
and finally in a third terminal:
```python -m src.real_time_noise_processing.main.run -f test\config\microphone_to_socket.yaml```

# Implementation details

- I decided to use the python package sounddevice to capture and play audio.
  - To see the documentation of this package, see  https://python-sounddevice.readthedocs.io/en/0.4.1/
  - The reason I chose this library, is that it can store the audio it records,
      and play audio, from numpy arrays directly, in addition to being able to
      work with bytes (and maybe even streams).
  - Another reason I chose this library, is that it is cross-platform.

# Things to consider
- Consider changing the reader/writer design to a more general observer design (use https://refactoring.guru/design-patterns/observer/python/example for a good example of how to do this correctly)

# TODO
- [x] Change class names from Receiver to Reader and Listener to Writer
- [x] Move code to separate files outside of POC
- [x] Add CLI script to run the different modes of operation
- [x] Add option to make the wait function of SocketWriter not block
- [x] Create FileReader (FilePlayer in class diagram)
- [x] Create FileWriter (SaveToFile in class diagram)
- [x] Create Processor Writer (Gets a Processor and a Writer, and puts everything it gets through the processor and sends its output to the Writer)
- [x] Create Splitter Writer (can be created by a Splitter Processor)
- [x] Create Player Writer
- [x] Add finalize() call to all writers and processors, to tell them that the input is done and let them finish
- [x] Change the initialization of speaker player to only start after we get the first bit of data to the writer, so it will not self terminate before it got any data.
- [x] Create requirements.txt with sections for visualizer / sounddevice / etc
- [x] Add a call such as writer.finish for when there is no more data to read (like when a socket is closed). This will give the writer time to finalize anything it needs (like send all the data it has buffered, or display a message, or close a file).
- [x] Create Noise Reduction Processor
- [ ] Create a wav file reader and writer (or add the needed flags and code to the current file reader and writer), using the example of play_long_file from sounddevice
- [ ] Change the interface between the readers and the writers to use numpy array of floats instead of a raw data array. This will require a change of all the objects that use sounddevice to use the numpy stream instead of the raw stream, and require every object that currently uses a bytearray of raw data to use a numpyarray of floats instead.
- [ ] Add utils module and implement function to convert from data to samples array and back, and change the implementation of all the processors and writers that do this themselves to use this (audio_visualizer, multiplier, dccrn_processor)
- [ ] Make the run script easier to run (require less dots in the name - by exposing it directly from the __init__.py script in the src folder, with a good name)
- [ ] Add docstring to everything in the code
- [ ] Add details and classes to class diagram
- [ ] Create sequence diagram
- [ ] Update README with diagrams and startup instructions
- [ ] Improve the DCCRN processor so that it will use overlapping windows instead of one window, to reduce problems that are caused at the end of the windows.
- [ ] Add checks when importing libraries to not fail if a certain library does not exist, and instead just continue and not support the reader/processor/writer that uses this library (maybe even expose some way for the objects to tell which libraries they need, to know if they can be used before actually running them?)
- [ ] Add some kind of buffering to reading the data in the readers (read some amount of data before starting to pass the data to the writer)
- [ ] Remove audio files from the repo
- [ ] Change abstract classes to really be abstract (or interfaces if possible)
- [ ] standardize arguments to the readers and writers (and replace additional_args with sounddee_args), and document somewhere what arguments exist in all of the classes
- [ ] Add some way for the reader to tell the writer what samplerate (and maybe data type and such) it works in, so that you can set the samplerate only in the reader and it will propagate to the processors and writers, and so that if the reader can infer the samplerate (like in the case of the file reader), there will not be a need to specify a samplerate at all.
- [ ] Find a way to synchronize the audio of speaker_player and the video of audio_visualizer
- [ ] Tidy up code
- [ ] Make code robust (make sure nothing is hard coded, everything is according to specs, try on different systems, etc)
- [ ] Simplify gitignore
- [ ] Make into package (add setup.py, make imports relative, add tests, etc). Read https://blog.ionelmc.ro/2014/05/25/python-packaging/ and https://github.com/pypa/sampleproject to see what needs to be done.