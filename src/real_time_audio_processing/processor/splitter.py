#!/usr/bin/env python3

"""Processor to run split the data into multiple writers. This is used in order to have multiple writer with one reader.
"""
from __future__ import absolute_import

from .processor import Processor

class Splitter(Processor):
    """Call multiple writers."""
    def __init__(self, writers):
        """Initialize a Splitter processor.
        
        Args:
            writers (list):     List of writers to call each time this processor is called.
        """
        self.writers = writers

    def process(self, data):
        """Call each of the writers with the data.

        Args:
            data (buffer):        data to write. It is a buffer with length of blocksize*sizeof(dtype).
        """
        # Tell all the writers about the data
        for writer in self.writers:
            writer.data_ready(data)

    def wait(self):
        """Wait for each of the writers, and give the writers time to process the data

        Returns: True if one of the processors returns True.
        """
        status = False
        for writer in self.writers:
            status = writer.wait() or status
        return status

    def finalize(self):
        """Give the writers time to finalize any remaining calculation or free resources
        """
        for writer in self.writers:
            status = writer.finalize()