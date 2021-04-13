#!/usr/bin/env python3

"""Play audio data over speakers.
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
    def __init__(self, blocking_time=0.001, additional_args=None, max_empty_buffers=10, verbose=False):
        """Initialize a SpeakerPlayer object.
        This writer works by using the sound device library to write to a device.

        This writer opens a new thread which plays the data, so it does not have to block.
        Args:
            blocking_time (float):      Time to block each time wait is called.
            additional_args (dict):     Additional arguments to pass to the creation of the sound device stream.
            max_empty_buffers (int):    Maximum amount of empty buffers to tolerate before closing the writer.
            verbose (bool):             Should each minor error be printed.
        """
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
        """Start the stream of audio out.
        """
        def audio_callback(outdata, frames, timings, status):
            """Callback for the sound device stream, to be called from a separate thread for each audio block.
            This callback must have this exact interface in order to work with sounddevice stream.

            Args:
                outdata (buffer):        Data to process. It is a buffer of frames*sizeof(dtype).
                    by default sizeof(dtype) is 8.
                frames (int):           Number of samples to process. Should be the same as blocksize.
                timings (CData):        Time of the samples in outdata. (from what I saw it is always 0).
                status (CallbackFlags): Status of the stream. (were there dropped buffers, etc).

            """
            if status:
                # Print the status
                if self.verbose:
                    print("SpeakerPlayer callback status:", status, file=sys.stderr)
                # We did not manage to ready the data in time, zero the buffer.
                # TODO: Check if this is really needed, or if it only hinders the code.
                if status.output_underflow:
                    outdata[:] = b"\x00"*len(outdata)
                    return
            try:
                # Get the next block from the queue
                data = self.q.get_nowait()
                self.empty_buffer_count = 0
            except queue.Empty as e:
                # Print the error that no data was ready in time
                if self.verbose:
                    print('SpeakerPlayer queue is empty', file=sys.stderr)
                if self.empty_buffer_count >= self.max_empty_buffers:
                    # Too many empty buffers in a row, time to abort
                    raise sd.CallbackAbort from e
                else:
                    # We are still not in the maximum amount of allowed empty buffers, so just zero the buffer
                    self.empty_buffer_count += 1
                    outdata[:] = b"\x00"*len(outdata)
                    return
            outdata[:] = data

        # Open the output stream
        self.stream = sd.RawOutputStream(callback=audio_callback, **self.additional_args, channels=1, finished_callback=self.event.set)

    def data_ready(self, data):
        """Add the data to the queue, so audio_callback will output it when called.

        Args:
            data (buffer):        data to write. It is a buffer with length of blocksize*sizeof(dtype).
        """
        self.q.put(data)
        # Raise a flag that we got the first bit of data
        self.did_get_first_data = True

    def wait(self):
        """Wait for the determined amount of time, or for a keyboard interrupt.

        Returns:
            True if there was a keyboard interrupt, False otherwise.
        """
        # if we didn't get any data yet, just return and wait for data to arrive.
        if not self.did_get_first_data:
            return False
        elif not self.did_start_playing:
            # Start the stream after we get the first data to output
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
        """Wait for the output stream to finish.
        """
        print("waiting for stream to finish")
        self.event.wait()
