#!/usr/bin/env python3

""" POC for capturing audio and immediatly playing it.
"""

import sounddevice as sd

def echo_callback(indata, outdata, frames, time, status):
    if status:
        print(status)
    outdata[:] = indata

def capture_and_play(duration, use_numpy, sample_rate=None):
    additional_args = {}
    # Decide if we want to use numpy or raw bytes.
    if use_numpy:
        stream = sd.Stream
    else:
        stream = sd.RawStream
        additional_args["dtype"] = "int24"
    # Add a specific sample rate.
    if sample_rate is not None:
        additional_args["samplerate"] = sample_rate
    with stream(channels=1, callback=echo_callback, **additional_args):
        sd.sleep(int(duration * 1000))

def main():
    duration = 20 # seconds
    use_numpy = True
    #  sample_rate = 16000 # This one does not work!
    #  sample_rate = 48000
    #  sample_rate = 44100
    sample_rate = None
    capture_and_play(duration, use_numpy, sample_rate)

if '__main__' == __name__:
    main()

