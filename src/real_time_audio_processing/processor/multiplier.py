#!/usr/bin/env python3

"""Processor to multiply the samples processed by some factor.
This can be used to increase or decrease the volume of the stream.
This processor is mostly here as an example of what a processor can do and how to write one.
"""

from __future__ import absolute_import

from .processor import Processor

from ..utils.raw_samples_converter import raw_samples_to_array, array_to_raw_samples

class Multiplier(Processor):
    """Multiply each sample by a given factor.
    """
    def __init__(self, factor, sample_size=4):
        """Initialize a Multiplier processor.
        
        Args:
            factor (float):     Factor to multiply each sample by.
            sample_size (int):  Size of each sample of raw bytes. Used in the conversion from the raw bytes to the
                actual sample value.
        """
        self.factor = factor
        self.sample_size = sample_size

    def process(self, data):
        """Multiply each of the samples.

        Args:
            data (buffer):        data to multiply. It is a buffer with length of blocksize*sizeof(dtype).
        """
        # Convert the raw data to actual samples
        samples = raw_samples_to_array(data, self.sample_size)
        # Multiply each of the samples
        samples = [sample * self.factor for sample in samples]
        # Convert back the multiplied samples to raw data
        array_to_raw_samples(samples, data, self.sample_size)

    def wait(self):
        """Always return False, to never finish.
        """
        return False

    def finalize(self):
        """Do nothing, as all the processing was done in the process function.
        """
        return