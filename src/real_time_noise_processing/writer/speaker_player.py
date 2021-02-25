#!/usr/bin/env python3

""" Write audio data to a socket.
"""

from __future__ import absolute_import

from .writer import Writer
import sounddevice as sd
import sys
import queue
import time


class SpeakerPlayer(Writer):
    """Get audio data from a reader and play it on speakers"""
    def __init__(self, timeout=None, additional_args=None):
        self.timeout = timeout

        self.additional_args = {}
        if additional_args is not None:
            self.additional_args = additional_args

        self.q = queue.Queue()

        # Start playback
        self.start_stream()

    def start_stream(self):

        def audio_callback(outdata, frames, time, status):
            if status.output_underflow:
                print('Output underflow: increase blocksize?', file=sys.stderr)
                return
            try:
                data = self.q.get_nowait()
            except queue.Empty as e:
                print('Buffer is empty: increase buffersize?', file=sys.stderr)
                return
            outdata[:] = data

        self.stream = sd.RawOutputStream(callback=audio_callback, **self.additional_args, channels=1)

    def data_ready(self, data):
        """Add the audio samples data to queue, to be consumed by update_plot every time we need to update the plot.
        This function converts the audio to numpy array before adding it to the queue.

        Args:
            data (buffer):        data to write. It is a buffer with length of blocksize*sizeof(dtype).
        """
        self.q.put(data)

    def wait(self):
        with self.stream:
            while True:
                time.sleep(10)
        return True
