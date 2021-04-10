#!/usr/bin/env python3

"""Write audio data to an output device.
"""

from __future__ import absolute_import

from abc import ABC, abstractmethod

class Writer(ABC):
    """Abstract writer class to write audio data gotten from a reader."""

    @abstractmethod
    def data_ready(self, data):
        """Called by a reader for every block of data. 
        Process in this function should be done in minimal amount of time, to allow the reader to work in real time.
        A common practice for this function is to write the data gotten in this function to a queue, which wait()
        can later process.

        Note that this function might be called from a different thread, which is time critical, and if not written
        carefully, this function might cause deadlocks or other problems because it runs from a different thread
        than wait().

        Args:
            data (buffer):        data to write. It is a buffer with length of blocksize*sizeof(dtype). To replace 
                a byte in this buffer, use ```data[start_index:end_index]=X``` where X is a bytes object.
        """
        pass

    @abstractmethod
    def wait(self):
        """Process data gotten in data_ready.
        This function is used by the reader in order to know when it should stop reading, and in order to
        give this writer time to process the data in the main thread.
        If the reader reads input in a different thread, this function can block, and return with True
        when it wants to stop writing data. If the reader reads input in the main thread, this function
        should only block for a minimal amount of time, and let the reader read more data.

        Returns:
            True if we are done processing all the data and wish to close the reader, False otherwise.
            Note that you should only return True when you never want to get more data to this writer.
        """
        pass

    @abstractmethod
    def finalize(self):
        """Process any leftover data, and close the writer.
        This function is only called after all the available data was sent to the writer with calls
        to data_ready, and the reader is done and wants to finish running.
        In this function the writer should finish processing any leftover data it saved for itself,
        clean and close any leftover resources, and once it returns the program will be closed.
        This function is called in the same thread as the wait function, and the writer can take as long 
        as it needs to finish processing the data.
        This function will be called even if the writer chooses to finish the run by returning True in
        the wait function.
        """
        pass