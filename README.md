# real_time_speech_denoiser
Real time application to show speech denoising in action.

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
- Consider adding a call such as writer.finish for when there is no more data to read (like when a socket is closed). This will give the writer time to finalize anything it needs (like send all the data it has buffered, or display a message, or close a file).
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
- [ ] Find a way to synchronize the audio of speaker_player and the video of audio_visualizer
- [ ] Create sequence diagram
- [ ] Tidy up code
- [ ] Add docstring to everything in the code
- [ ] Add details and classes to class diagram
- [ ] Update README with diagrams and startup instructions
- [ ] Create requirements.txt with sections for visualizer / sounddevice / etc
- [ ] Simplify gitignore
- [ ] Make code robust (make sure nothing is hard coded, everything is according to specs, try on different systems, etc)
- [ ] Change abstract classes to really be abstract (or interfaces if possible)
- [ ] Make into package (add setup.py, make imports relative, add tests, etc). Read https://blog.ionelmc.ro/2014/05/25/python-packaging/ and https://github.com/pypa/sampleproject to see what needs to be done.
- [ ] Create Noise Reduction Processor in a different project
