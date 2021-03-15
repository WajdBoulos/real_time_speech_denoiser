#!/usr/bin/env python3

from __future__ import absolute_import

from .reader import Reader

class FileReader(Reader):
    def __init__(self, writer, path, blocksize, sample_size=4):
        self.writer = writer
        self.path = path
        self.blocksize = blocksize
        self.sample_size = sample_size
        self.initialize_file()

    def initialize_file(self):
        print("opening input file")
        self.file = open(self.path, "rb")


    def read(self):
        while not self.writer.wait():
            data = self.file.read(self.blocksize * self.sample_size)
            if len(data) != (self.blocksize * self.sample_size):
                print("reached end of input file")
                break
            data = bytearray(data)
            self.writer.data_ready(data)
        print("closing input file")
        self.file.close()
        self.writer.finalize()