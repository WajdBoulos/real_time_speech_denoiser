#!/usr/bin/env python3

""" Write audio data to a file.
"""

from __future__ import absolute_import

from .writer import Writer
import time
import soundfile as sf

class FileWriter(Writer):
    """
    @note The file created by this writer is not a wav file, it is a file with the raw samples only.
    """
    def __init__(self, path, wav_format=True, blocking_time=0.1, samplerate=16000, sample_size=4):
        self.path = path
        self.wav_format = wav_format
        self.blocking_time = blocking_time
        self.samplerate = samplerate
        self.sample_size = sample_size
        self.initialize_file()

    def initialize_file(self):
        print("opening output file, send ctrl-c to stop")
        self.file = open(self.path, "wb")
        if self.wav_format:
            self.sf = sf.SoundFile(self.file, "wb", channels=1, samplerate=self.samplerate)
            if self.sample_size == 4:
                self.dtype = "float32"
            elif self.sample_size == 2:
                self.dtype = "int16"
            else:
                raise ValueError(f"unsupported sample size {self.sample_size}")

    def data_ready(self, data):
        if self.wav_format:
            self.sf.buffer_write(data, self.dtype)
        else:
            self.file.write(data)

    def wait(self):
        try:
            time.sleep(self.blocking_time)
        except KeyboardInterrupt:
            return True
        return False

    def finalize(self):
        if self.file is not None:
            print("closing output file")
            if self.wav_format:
                self.sf.close()
            self.file.close()
            self.file = None
