#!/usr/bin/env python3

"""Read audio data from a microphone.
"""
from __future__ import absolute_import

from .reader import Reader
import sounddevice as sd
import sys

class MicrophoneReader(Reader):
    """Read audio from the microphone and send each block of samples to a writer.
    """
    def __init__(self, writer, additional_args=None, verbose=False):
        """Initialize a MicrophoneReader object.
        This reader works by using the sound device library to listen to a device.

        This reader opens a different thread to read its data, so the writer can block when writer.wait is called.

        Args:
            writer (Writer):        Writer object to give the data to.
            additional_args (dict): Additional arguments to pass to the creation of the sound device stream.
            verbose (bool):         Should each minor error be printed.
        """
        self.writer = writer
        self.verbose = verbose
        self.additional_args = {}
        if additional_args is not None:
            self.additional_args = additional_args

        # Use a numpy array stream
        self.stream = sd.InputStream

    def read(self):
        """Read from a microphone, and send it to a writer.

        This function initializes a different thread to record audio. 
        This means that the writer can block when writer.wait is called.
        """
        def audio_callback(indata, frames, time, status):
            """Callback for the sound device stream, to be called from a separate thread for each audio block.
            This callback must have this exact interface in order to work with sounddevice stream.

            Args:
                indata (buffer):        Data to process. It is a buffer of frames*sizeof(dtype).
                    by default sizeof(dtype) is 8.
                frames (int):           Number of samples to process. Should be the same as blocksize.
                time   (CData):         Time of the samples in indata. (from what I saw it is always 0).
                status (CallbackFlags): Status of the stream. (were there dropped buffers, etc).

            """
            if status and self.verbose:
                print("MicrophoneReader callback status:", status, file=sys.stderr)
            self.writer.data_ready(indata.reshape((len(indata))))

        # Initialize a stream for input.
        stream = self.stream(callback=audio_callback, **self.additional_args, channels=1)
        
        # Open the stream and start to record audio
        print("opening input stream")
        with stream:
            # Wait for the writer to tell us we need to stop recording
            while not self.writer.wait():
                pass
        print("done with input stream")

        # Let the writer do any final processing before exiting
        self.writer.finalize()
