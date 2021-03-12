#!/usr/bin/env python3

""" Show GUI window with visualization of audio data.
"""
from __future__ import absolute_import

from .writer import Writer
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import queue
import struct

class AudioVisualizer(Writer):
    """GUI Visualizer for audio data"""
    def __init__(self, samplerate, duration=200.0, interval=30.0, downsample=1, blocking_time=None, blocksize=1024, sample_size=4):
        """Initialize an AudioVisualizer object.
        This writer opens a GUI window, and shows an interactive visualization of the audio data it gets in real time.

        The wait function of this object can be blocking or non-blocking, based on the blocking_time argument.

        Args:
            samplerate (int):       Samplerate of the audio. Used to synchronize the speed of the visualized data to
                the audio data received.
            duration (float):       Amount of time (in seconds) to show in the window.
            interval (float):       Delay (in milliseconds) between frames of the animation of the visualization.
            downsample (int):       Amount of samples to skip for every sample shown on the animation. Useful when
                sampling in very high frequencies.
            blocking_time (float):  Maximum amount of time to block each time wait() is called. If None, wait() will
                block until the visualization window is closed.
            blocksize (int):        Amount of samples in each block that will be received in data_ready.
            sample_size (int):      Amount of bytes in each sample.

        """
        self.samplerate = samplerate
        self.duration = duration
        self.interval = interval
        self.downsample = downsample
        self.blocking_time = blocking_time
        if self.blocking_time is None:
            self.blocking = True
        else:
            self.blocking = False
        self.did_show = False
        self.blocksize = blocksize
        self.sample_size = sample_size

        self.initialize_parameters()

        self.q = queue.Queue()
        self.initialize_animation()

    def initialize_parameters(self):
        if self.sample_size == 4:
            self.unpack_string = "f"
            self.data_range = (-1, 1)
        elif self.sample_size == 2:
            self.unpack_string = "h"
            self.data_range = (-32768, 32767)
        else:
            raise ValueError(f"unsupported sample size {self.sample_size}")

    def initialize_animation(self):
        """Initialize parameters for the visualization animation window and set its update function.
        """
        self.length = int(self.duration * self.samplerate / (self.downsample))
        self.plotdata = np.zeros((self.length, 1))
        fig, ax = plt.subplots()
        self.lines = ax.plot(self.plotdata)

        # Magic to start an animation. Copied from sounddevice plot_input example.
        ax.axis((0, len(self.plotdata), *self.data_range))
        ax.set_yticks([0])
        ax.yaxis.grid(True)
        ax.tick_params(bottom=False, top=False, labelbottom=False,
                       right=False, left=False, labelleft=False)
        fig.tight_layout(pad=0)

        def update_plot_wrapper(frame):
            """Wrapper for the update_plot function, to make it fit the interface known by FuncAnimation.
            """
            return self.update_plot(frame)

        self.ani = FuncAnimation(fig, update_plot_wrapper, interval=self.interval, blit=True)

    def data_ready(self, data):
        """Add the audio samples data to queue, to be consumed by update_plot every time we need to update the plot.
        This function converts the audio to numpy array before adding it to the queue.

        Args:
            data (buffer):        data to write. It is a buffer with length of blocksize*sizeof(dtype).
        """
        old_data = data
        data = np.zeros((self.blocksize, 1), dtype=np.float32)
        for i in range(0, len(old_data), self.sample_size):
            data[i//self.sample_size] = struct.unpack(self.unpack_string, old_data[i:i+self.sample_size])[0]

        self.q.put(data[::self.downsample])

    def wait(self):
        """Show the visualization window in the first run, and allow it to update for blocking_time seconds on each
        later run. If self.blocking is True, this function will only return from its first call when the animation
        window will be closed.

        Returns:
            True if the animation window was closed by the user, False otherwise.
        """
        if not self.did_show:
            plt.show(block=self.blocking)
            self.did_show = True
        else:
            plt.pause(self.blocking_time)
        if plt.get_fignums():
            return False
        return True

    def update_plot(self, frame):
        """Callback function for the animation.
        Get all the data from the queue filled by data_ready, and update the animation accordingly.
        """
        while True:
            try:
                data = self.q.get_nowait()
            except queue.Empty:
                break
            shift = len(data)
            self.plotdata = np.roll(self.plotdata, -shift, axis=0)
            self.plotdata[-shift:, :] = data
        for column, line in enumerate(self.lines):
            line.set_ydata(self.plotdata[:, column])
        return self.lines
