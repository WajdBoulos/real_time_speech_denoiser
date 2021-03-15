#!/usr/bin/env python3

""" Read audio data from an input device and send it to be written
"""
from __future__ import absolute_import

class Reader(object):
    """Abstract reader class to read audio from any input device"""

    def read(self):
        """Read input data, and send it to a writer. 
        This function only returns when the writer signals that it does not want any more data.
        This funciton calls writer.data_ready for every block of data it reads.
        This function calls writer.wait in order to give the writer time to process the data, 
        and only return once writer.wait returns True.
        This function can get its data in one of two ways. It can get its data in another therad, 
        in which case it should only initialize the other thread and then call writer.wait in a loop
        until it returns True (and in this case writer.wait is allowed to block for unlimited time).
        Or this function can get its data in the main therad, in which case it should get the data
        in this function, call writer.data_ready, and then call writer.wait to give the writer time
        to process the new data (and in this case writer.wait should not block for more than a minimal
        amount of time - 10 to 100 ms should be fine).
        If the data the reader reads has some maximal size (be it a finite file, a socket that is closed,
        or the user asks that the reader finish), after all the data was sent to the writer with calls
        to writer.data_ready, the reader must call writer.finalize to notify the writer that no new data
        is about to arrive. The call to finalize should be made from the same thread as the calls to
        writer.wait.
        The data passed to writer.data_ready should be of a type that is compatible to a bytearray,
        so that the line ```data[0:1]=b"a"``` will replace the first byte with the byte 'a'.
        """
        pass
