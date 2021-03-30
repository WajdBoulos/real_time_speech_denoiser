#!/usr/bin/env python3

from __future__ import absolute_import

from .reader import Reader
import soundfile as sf

class FileReader(Reader):
    def __init__(self, writer, path, blocksize, wav_format=True, sample_size=4):
        self.writer = writer
        self.path = path
        self.blocksize = blocksize
        self.sample_size = sample_size
        self.wav_format = wav_format
        self.initialize_file()

    def initialize_file(self):
        print("opening input file")
        self.file = open(self.path, "rb")
        if self.wav_format:
            self.sf = sf.SoundFile(self.file, "rb")
            if self.sample_size == 4:
                self.dtype = "float32"
            elif self.sample_size == 2:
                self.dtype = "int16"
            else:
                raise ValueError(f"unsupported sample size {self.sample_size}")


    def read(self):
        while not self.writer.wait():
            if self.wav_format:
                data = self.sf.buffer_read(self.blocksize, dtype=self.dtype)
            else:
                data = self.file.read(self.blocksize * self.sample_size)
            if len(data) != (self.blocksize * self.sample_size):
                print("reached end of input file")
                break
            data = bytearray(data)
            self.writer.data_ready(data)
        print("closing input file")
        if self.wav_format:
            self.sf.close()
        self.file.close()
        self.writer.finalize()