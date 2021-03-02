#!/usr/bin/env python3

""" Process audio data.
"""
from __future__ import absolute_import

class Processor(object):
    """Abstract processor class to process audio data.
    This class was created to fill a gap between a reader and a writer, in order to add processing stages
    between reading the data and writing it. It is also useful for multiple processing stages, as multiple
    Processor Writers can be chained for more complex processing.
    """

    def process(self, data):
        """Process a block of data.
        A Processor Writer should call this function for every block of data.
        This function should always return data with the same size as the data it gets.
        
        Args:
            data (buffer):        data to process. It is a buffer with length of blocksize*sizeof(dtype).
        Returns:
            Nothing. changes the data in place inside the data buffer.
        """
        pass

    def wait(self):
        """Run time to process any data.
        A Processor Writer should call this function when it is okay to take more time to run.

        A processor should not use the process function for a long time, and should do most of the heavy
        processing in this function instead. Note that there is no guarantee about the frequency of calls
        to this function in relation to the process function.
        
        Returns:
            True if we need to stop the program, False otherwise.
        """
        pass
