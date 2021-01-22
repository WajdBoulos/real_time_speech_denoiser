#!/usr/bin/env python3

""" Read audio data from an input device and send it to be written
"""

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
        """
        pass
