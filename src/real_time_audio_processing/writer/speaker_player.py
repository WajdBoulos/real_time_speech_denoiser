#!/usr/bin/env python3

""" Write audio data to a socket.
"""

from __future__ import absolute_import

from .writer import Writer
import sounddevice as sd
import sys
import queue
import time

import threading

class SpeakerPlayer(Writer):
    """Get audio data from a reader and play it on speakers"""
    def __init__(self, blocking_time=0.1, additional_args=None, max_empty_buffers=10, verbose=False):
        self.additional_args = {}
        if additional_args is not None:
            self.additional_args = additional_args

        self.q = queue.Queue()
        self.event = threading.Event()
        self.empty_buffer_count = 0
        self.max_empty_buffers = max_empty_buffers
        self.did_get_first_data = False

        self.did_start_playing = False
        self.blocking_time = blocking_time
        self.verbose = verbose

        # Start playback
        self.start_stream()

    def start_stream(self):

        def audio_callback(outdata, frames, timings, status):
            if status:
                if self.verbose:
                    print("SpeakerPlayer callback status:", status, file=sys.stderr)
                if status.output_underflow:
                    outdata[:] = b"\x00"*len(outdata)
                    return
            try:
                data = self.q.get_nowait()
                self.empty_buffer_count = 0
            except queue.Empty as e:
                if self.verbose:
                    print('Buffer is empty', file=sys.stderr)
                if self.empty_buffer_count >= self.max_empty_buffers:
                    # Too many empty buffers in a row, time to abort
                    raise sd.CallbackAbort from e
                else:
                    self.empty_buffer_count += 1
                    outdata[:] = b"\x00"*len(outdata)
                    return
            outdata[:] = data

        self.stream = sd.RawOutputStream(callback=audio_callback, **self.additional_args, channels=1, finished_callback=self.event.set)

    def data_ready(self, data):
        self.q.put(data)
        self.did_get_first_data = True

    def wait(self):
        if not self.did_get_first_data:
            return False
        elif not self.did_start_playing:
            self.did_start_playing = True
            print("opening output stream, send ctrl-c to stop")
            self.stream.__enter__()
        else:
            # Sleep, and exit if we get a keyboard interrupt
            try:
                time.sleep(self.blocking_time)
            except KeyboardInterrupt:
                print("closing output stream")
                self.stream.__exit__()
                return True
        return False

    def finalize(self):
        print("waiting for stream to finish")
        self.event.wait()
