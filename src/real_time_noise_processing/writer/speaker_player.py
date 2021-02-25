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
    def __init__(self, timeout=0.1, additional_args=None,):
        self.additional_args = {}
        if additional_args is not None:
            self.additional_args = additional_args

        self.q = queue.Queue()

        self.did_start_playing = False
        self.timeout = timeout

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
        self.q.put(data)

    def wait(self):
        if not self.did_start_playing:
            self.did_start_playing = True
            print("opening output stream, send ctrl-c to stop")
            self.stream.__enter__()
        else:
            # Sleep, and exit if we get a keyboard interrupt
            try:
                time.sleep(self.timeout)
            except KeyboardInterrupt:
                print("closing output stream")
                self.stream.__exit__()
                return True
        return False
