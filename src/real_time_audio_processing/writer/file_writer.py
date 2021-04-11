#!/usr/bin/env python3

"""Write audio data to a file.
"""

from __future__ import absolute_import

from .writer import Writer
import time
import soundfile as sf

class FileWriter(Writer):
    """Write audio to a file.
    """
    def __init__(self, path, blocking_time=0.001, samplerate=16000):
        """Initialize a FileWriter object.
        This writer works by using the soundfile library to open a wav file and write the data to it.

        Args:
            path (str):             Path to the file to write. If it already exists, it will truncated.
            blocking_time (float):  Time to block each time the wait method is called.
            samplerate (int):       Samplerate to write the data to the file in.
        """
        self.path = path
        self.blocking_time = blocking_time
        self.samplerate = samplerate
        self.initialize_file()

    def initialize_file(self):
        """Open the file to write to and ready it for writing.
        """
        # Open the file
        print("opening output file, send ctrl-c to stop")
        self.file = open(self.path, "wb")

        # Start a sound file object form the file
        self.sf = sf.SoundFile(self.file, "wb", channels=1, samplerate=self.samplerate)

    def data_ready(self, data):
        """Write audio data to the file

        Args:
            data (buffer):        data to write. It is a buffer with length of blocksize.
        """
        self.sf.write(data)

    def wait(self):
        """Wait for the determined amount of time, or for a keyboard interrupt.

        Returns:
            True if there was a keyboard interrupt, False otherwise.
        """
        try:
            time.sleep(self.blocking_time)
        except KeyboardInterrupt:
            return True
        return False

    def finalize(self):
        """Close the file (which flushes any unwritten data to it).
        """
        if self.file is not None:
            print("closing output file")
            self.sf.close()
            self.file.close()
            self.file = None
