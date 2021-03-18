#!/usr/bin/env python3

""" Write audio data to a file.
"""

from __future__ import absolute_import

from .writer import Writer
import time

class FileWriter(Writer):
    """
    @note The file created by this writer is not a wav file, it is a file with the raw samples only.
    """
    def __init__(self, path, blocking_time=0.1):
        self.path = path
        self.blocking_time = blocking_time
        self.initialize_file()

    def initialize_file(self):
        print("opening output file, send ctrl-c to stop")
        self.file = open(self.path, "wb")

    def data_ready(self, data):
        self.file.write(data)

    def wait(self):
        try:
            time.sleep(self.blocking_time)
        except KeyboardInterrupt:
            print("closing output file")
            self.file.close()
            self.file = None
            return True
        return False

    def finalize(self):
        if self.file is not None:
            self.file.close()
            self.file = None
