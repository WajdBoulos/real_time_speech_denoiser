# real_time_speech_denoiser
Real time application to show speech denoising in action.

# Implementation details

- I decided to use the python package sounddevice to capture and play audio.
  - To see the documentation of this package, see  https://python-sounddevice.readthedocs.io/en/0.4.1/
  - The reason I chose this library, is that it can store the audio it records,
      and play audio, from numpy arrays directly, in addition to being able to
      work with bytes (and maybe even streams).
  - Another reason I chose this library, is that it is cross-platform.
