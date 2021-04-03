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
    def __init__(self, path, wav_format=True, blocking_time=0.001, samplerate=16000, sample_size=4):
        """Initialize a FileWriter object.
        This writer works by using the soundfile library to open a wav file and write the data to it.

        Args:
            path (str):             Path to the file to write. If it already exists, it will truncated.
            wav_format (bool):      Is the file a wav format. If it is not, it is assumed that the file is just a blob of
                raw samples.
            blocking_time (float):  Time to block each time the wait method is called.
            samplerate (int):       Samplerate to write the data to the file in.
            sample_size (int):      The size in bytes of each sample when formatted as raw data.

        """
        self.path = path
        self.wav_format = wav_format
        self.blocking_time = blocking_time
        self.samplerate = samplerate
        self.sample_size = sample_size
        self.initialize_file()

    def initialize_file(self):
        """Open the file to write to and ready it for writing.
        """
        # Open the file
        print("opening output file, send ctrl-c to stop")
        self.file = open(self.path, "wb")
        if self.wav_format:
            # Start a sound file object form the file
            self.sf = sf.SoundFile(self.file, "wb", channels=1, samplerate=self.samplerate)
            # Decide on the type of data to convert the samples into when writing it
            if self.sample_size == 4:
                self.dtype = "float32"
            elif self.sample_size == 2:
                self.dtype = "int16"
            else:
                raise ValueError(f"unsupported sample size {self.sample_size}")

    def data_ready(self, data):
        """Write audio data to the file

        Args:
            data (buffer):        data to write. It is a buffer with length of blocksize*sizeof(dtype).
        """
        if self.wav_format:
            self.sf.buffer_write(data, self.dtype)
        else:
            self.file.write(data)

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
            if self.wav_format:
                self.sf.close()
            self.file.close()
            self.file = None
