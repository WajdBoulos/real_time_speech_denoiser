#!/usr/bin/env python3

"""Process audio data.
"""
from __future__ import absolute_import

from abc import ABC, abstractmethod

class Processor(ABC):
    """Abstract processor class to process audio data.
    This class was created to fill a gap between a reader and a writer, in order to add processing stages
    between reading the data and writing it. It is also useful for multiple processing stages, as multiple
    Processor Writers can be chained for more complex processing.
    """

    @abstractmethod
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

    @abstractmethod
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

    @abstractmethod
    def finalize(self):
        """Process any leftover data, and close the processor.
        This function is only called after all the available data was sent to the processor with calls
        to process, and the reader is done and wants to finish running.
        In this function the processor should finish processing any leftover data it saved for itself,
        clean and close any leftover resources, and once it returns the program will be closed.
        This function is called in the same thread as the wait function, and the processor can take as long 
        as it needs to finish processing the data.
        This function will be called even if the procesor chooses to finish the run by returning True in
        the wait function.
        """
        pass