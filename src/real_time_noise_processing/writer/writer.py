#!/usr/bin/env python3

""" Write audio data to an output device.
"""

class Writer(object):
    """Abstract writer class to write audio data gotten from a reader."""

    def data_ready(self, data):
        """Called by a reader for every block of data. 
        Process in this function should be done in minimal amount of time, to allow the reader to work in real time.
        A common practice for this function is to write the data gotten in this function to a queue, which wait()
        can later process.

        Note that this function might be called from a different thread, which is time critical, and if not written
        carefully, this function might cause deadlocks or other problems because it runs from a different thread
        than wait().

        Args:
            data (buffer):        data to write. It is a buffer with length of blocksize*sizeof(dtype).
        """
        pass

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
