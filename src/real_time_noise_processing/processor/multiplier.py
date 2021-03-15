#!/usr/bin/env python3

from __future__ import absolute_import

from .processor import Processor
import struct

class Multiplier(Processor):
    def __init__(self, factor, sample_size=4):
        self.factor = factor
        self.sample_size = sample_size

    def process(self, data):
        for i in range(0, len(data), self.sample_size):
            sample = struct.unpack('f', data[i:i+self.sample_size])[0]
            sample *= self.factor
            sample_bytes = struct.pack('f', sample)
            for j, value in enumerate(sample_bytes):
                data[i + j:i + j + 1] = bytes([value])

    def wait(self):
        # Never finish
        return False

    def finalize(self):
        # Nothing to do here
        return