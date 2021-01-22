#!/usr/bin/env python3

""" Process audio data.
"""

class Processor(object):
    """Abstract processor class to process audio data.
    This class was created to fill a gap between a reader and a writer, in order to add processing stages
    between reading the data and writing it. It is also useful for multiple processing stages, as multiple
    Processor Writers can be chained for more complex processing.
    """

    def Process(self, data):
        """Process a block of data.
        A Processor Writer should call this function for every block of data.
        This function should always return data with the same size as the data it gets.
        
        Args:
            data (buffer):        data to process. It is a buffer with length of blocksize*sizeof(dtype).
        Returns:
            Processed data buffer. If this function returns None, it is assumed that the data was processed
            in place inside the buffer.
        """
        pass
