#!/usr/bin/env python3

from __future__ import absolute_import

from .processor import Processor

from ..utils.raw_samples_converter import raw_samples_to_array, array_to_raw_samples

class Multiplier(Processor):
    def __init__(self, factor, sample_size=4):
        self.factor = factor
        self.sample_size = sample_size

    def process(self, data):
        samples = raw_samples_to_array(data, self.sample_size)
        samples = [sample * self.factor for sample in samples]
        array_to_raw_samples(samples, data, self.sample_size)

    def wait(self):
        # Never finish
        return False

    def finalize(self):
        # Nothing to do here
        return