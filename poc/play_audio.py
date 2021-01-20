#!/usr/bin/env python3

""" POC of playing audio from python
"""

import sounddevice as sd
import pickle

def play_file(file_stream, sample_rate=None):
    additional_args = {}
    # We must have a sample rate, so default to the default sample rate.
    if sample_rate is None:
        sample_rate = sd.query_devices(kind="output")["default_samplerate"]
    additional_args["samplerate"] = sample_rate

    # Get the data from the file
    recording = pickle.load(file_stream)

    # Play the audio.
    sd.play(recording, sample_rate)

    # If we want to stop the recording at any point, we can call sd.stop()
    # sd.play returns immediately, so we need to wait for it to finish.
    sd.wait()



def main():
    file_to_read = "test.audio"
    sample_rate = 16000

    with open(file_to_read, "rb") as f:
        play_file(f, sample_rate)

if '__main__' == __name__:
        main()
