#!/usr/bin/env python3

""" POC of capturing audio from a microphone
"""

import sounddevice as sd
import pickle

def capture_to_file(file_stream, duration, use_numpy=False, sample_rate=None):
    additional_args = {}
    # Decide if we want to use numpy or raw bytes.
    if use_numpy:
        stream = sd.Stream
    else:
        stream = sd.RawStream
    # We must have a sample rate, so default to the default sample rate.
    if sample_rate is None:
        sample_rate = sd.query_devices(kind="input")["default_samplerate"]
    additional_args["samplerate"] = sample_rate

    # Record the audio.
    recording = sd.rec(int(duration * sample_rate), channels=2,
                       **additional_args)
    # sd.rec returns immediately, so we need to wait for it to finish.
    sd.wait()

    # Write the output of the recording to a file.
    pickle.dump(recording, file_stream)


def main():
    file_to_write_to = "test.audio"
    duration = 5 # Seconds
    use_numpy = False
    sample_rate = 16000

    with open(file_to_write_to, "wb") as f:
        capture_to_file(f, duration, use_numpy, sample_rate)

if '__main__' == __name__:
        main()

