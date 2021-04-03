#!/usr/bin/env python3

"""Processor to run multiple processors under one processors. This is used in order to have only one ProcessorWriter,
that can do multiple processing steps.
"""

from __future__ import absolute_import

from .processor import Processor

class Pipeline(Processor):
    """Call multiple processors in order."""
    def __init__(self, processors):
        """Initialize a Pipeline processor.
        
        Args:
            processors (list):     List of processors to call each time this processor is called.
        """
        self.processors = processors

    def process(self, data):
        """Call each of the processors in order. Each processor will get the data after the previous processors already
        changed it according to their wishes.

        Args:
            data (buffer):        data to process. It is a buffer with length of blocksize*sizeof(dtype).
        """
        # Let each of the processors process the data one after the other
        for processor in self.processors:
            processor.process(data)

    def wait(self):
        """Wait for each of the processors, and give the processors time to process the data

        Returns: True if one of the processors returns True.
        """
        status = False
        for processor in self.processors:
            status = processor.wait() or status
        return status

    def finalize(self):
        """Give the processors time to finalize any remaining calculation or free resources
        """
        for processor in self.processors:
            processor.finalize()
