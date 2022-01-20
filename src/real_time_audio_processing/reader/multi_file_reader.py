#!/usr/bin/env python3

"""Read audio data from a file.
"""

from __future__ import absolute_import

from .reader import Reader
import soundfile as sf
import glob
import os

class FileReader(Reader):
    """Read audio from a file and send each block of samples to a writer.
    """
    def __init__(self, writer, path, blocksize, wav_format=True, sample_size=4):
        """Initialize a FileReader object.
        This reader works by using the soundfile library to open a wav file and read the data from it.

        This reader works in the main thread only, so the writer can not block for too long when writer.wait is called.

        Args:
            writer (Writer):    Writer object to give the data to.
            path (str):         Path to the file to read.
            blocksize (int):    Size of each block of data to send to the writer. The reader will read this amount of
                samples from the file, put the bytes in a buffer, and call the writer with the block. If the total
                length of the file does not divide by this block size, any leftover data after the last full block will
                be ignored.
            wav_format (bool):  Is the file a wav format. If it is not, it is assumed that the file is just a blob of
                raw samples.
            sample_size (int):  The size in bytes of each sample when formatted as raw data.

        """
        self.writer = writer
        self.blocksize = blocksize
        self.sample_size = sample_size
        self.wav_format = wav_format
        self.wav_path_list = glob.glob(os.path.join(path, "*.wav"))

    def initialize_file(self, filepath):
        """Open the file to read from and ready it for reading.
        """
        # Open the file
        print("opening input file")
        self.file = open(filepath, "rb")
        if self.wav_format:
            # Start a sound file object form the file
            self.sf = sf.SoundFile(self.file, "rb")
            # Decide on the type of data to convert the samples into when reading them
            if self.sample_size == 4:
                self.dtype = "float32"
            elif self.sample_size == 2:
                self.dtype = "int16"
            else:
                raise ValueError(f"unsupported sample size {self.sample_size}")


    def read(self):
        """Read from a file, and send it to a writer.

        This function works in the main thread only. 
        This means that the writer can NOT block when writer.wait is called.
        """
        for wav_path in self.wav_path_list:
            wav_filename = os.path.split(wav_path)[-1]
            self.initialize_file(wav_path)
            self.writer.initialize_file(wav_filename)
            # Work as long as the writer wants more data
            while not self.writer.wait():
                # Read the audio data from the file
                if self.wav_format:
                    data = self.sf.buffer_read(-1, dtype=self.dtype)
                    self.blocksize = len(data)
                else:
                    data = self.file.read(self.blocksize * self.sample_size)

                # Check that we read a full block of data
                if len(data) != (self.blocksize * self.sample_size):
                    print("reached end of input file")
                    break

                # Convert the data to a bytearray to conform to the API
                data = bytearray(data)

                # Send the data to the writer
                self.writer.data_ready(data)

            # Close the file
            print("closing input file")
            if self.wav_format:
                self.sf.close()
            self.file.close()

            # Let the writer do any final processing before exiting
            self.writer.finalize()
