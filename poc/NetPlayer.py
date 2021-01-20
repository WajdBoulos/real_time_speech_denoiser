#!/usr/bin/env python3

""" POC for capturing audio in one process and immediately playing it on another.
"""

import sounddevice as sd
import argparse
import sys

# For data visualizer
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import queue
import struct

# For socket
import socket
import select

class Receiver(object):
    """Abstract receiver class to receive audio from a device"""
    def listen(self):
        pass

class Recorder(Receiver):
    """Record audio from the microphone and send each block of samples to a listener"""
    def __init__(self, listener, additional_args):
        self.listener = listener
        self.additional_args = additional_args

        # Use a raw python buffer stream (and not a numpy array)
        self.stream = sd.RawInputStream
        # self.stream = sd.InputStream

    def listen(self):
        def audio_callback(indata, frames, time, status):
            """This is called (from a separate thread) for each audio block.

            Args:
                indata (buffer):        data to process. It is a buffer of frames*sizeof(dtype).
                    by default sizeof(dtype) is 8.
                frames (int):           number of samples to process. Should be the same as blocksize.
                time   (CData):         time of the samples in indata. (from what I saw it is always 0).
                stauts (CallbackFlags): status of the stream. (were there dropped buffers, etc).

            """
            if status:
                print(status, file=sys.stderr)
            self.listener.data_ready(indata)

        stream = self.stream(callback=audio_callback, **self.additional_args, channels=1)
        print("opening stream")
        with stream:
            # Wait for the listener to tell us we need to stop recording
            while not self.listener.wait():
                pass

        print("done with stream")

class Listener(object):
    """Abstract listener class to listen for samples"""
    def data_ready(self, data):
        pass

    def wait(self):
        pass

class SocketSender(Listener):
    """Listen to samples from a recorder and send them over socket"""
    def __init__(self, dest, timeout=None):
        self.dest = dest
        self.timeout = timeout
        self.initialize_socket()

    def initialize_socket(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.dest))

    def data_ready(self, data):
        while data:
            if self.socket is None:
                break
            sent_len = self.socket.send(data)
            data = data[sent_len:]
            if sent_len == 0:
                print("sent 0 bytes")
                self.socket.close()
                self.socket = None

    def wait(self):
        # Because no data should be received in this socket,
        # this actually just waits for the other end of the socket to close
        if self.timeout is not None:
            select.select([self.socket], [], [], self.timeout)
        else:
            select.select([self.socket], [], [])
        self.socket.close()
        self.socket = None
        print("socket was ready to read")
        return True



class SocketReceiver(Receiver):
    """Receive audio from a socket and send each block of samples to a listener"""
    def __init__(self, listener, address, blocksize, typesize=4):
        self.listener = listener
        self.address = address
        self.blocksize = blocksize
        self.typesize = typesize
        self.initialize_socket()

    def initialize_socket(self):
        self.listening_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listening_socket.bind((self.address))
        self.listening_socket.listen(1)
        self.socket, remote_addr = self.listening_socket.accept()
        print("got connection from", remote_addr)
        self.listening_socket.close()

    def listen(self):
        while not self.listener.wait() and self.socket is not None:
            # Read the data from the socket in blocks
            current_data = []
            remaining_len = self.blocksize * self.typesize
            while remaining_len != 0 and self.socket is not None:
                current_data.append(self.socket.recv(remaining_len))
                remaining_len -= len(current_data[-1])
                if len(current_data[-1]) == 0:
                    self.socket.close()
                    self.socket = None
            total_data = b"".join(current_data)
            self.listener.data_ready(total_data)
        if self.socket is not None:
            self.socket.close()
            self.socket = None

class AudioVisualizer(Listener):
    """GUI Visualizer for audio data"""
    def __init__(self, samplerate, duration=200.0, interval=30.0, downsample=1, blocking_time=None, typesize=4):
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
        self.typesize = typesize
        self.q = queue.Queue()

        self.length = int(self.duration * self.samplerate / (1000 * self.downsample))
        self.plotdata = np.zeros((self.length, 1))
        fig, ax = plt.subplots()
        self.lines = ax.plot(self.plotdata)

        ax.axis((0, len(self.plotdata), -1, 1))
        ax.set_yticks([0])
        ax.yaxis.grid(True)
        ax.tick_params(bottom=False, top=False, labelbottom=False,
                       right=False, left=False, labelleft=False)
        fig.tight_layout(pad=0)

        def update_plot_wrapper(frame):
            return self.update_plot(frame)

        self.ani = FuncAnimation(fig, update_plot_wrapper, interval=self.interval, blit=True)

    def data_ready(self, data):
        
        old_data = data
        data = np.zeros((1024, 1), dtype=np.float32)
        for i in range(0, len(old_data), self.typesize):
            data[i//self.typesize] = struct.unpack('f', old_data[i:i+self.typesize])[0]

        self.q.put(data[::self.downsample])

    def wait(self):
        if not self.did_show:
            plt.show(block=self.blocking)
            self.did_show = True
        else:
            plt.pause(self.blocking_time)
        if plt.get_fignums():
            return False
        return True

    def update_plot(self, frame):
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


def parse_arguments():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        '-l', '--list-devices', action='store_true',
        help='show list of audio devices and exit')
    args, remaining = parser.parse_known_args()
    if args.list_devices:
        print(sd.query_devices())
        parser.exit(0)
    parser = argparse.ArgumentParser(
        description=__doc__,
        parents=[parser])

    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument(
        '-r', '--recorder', action='store_true', help='open in recorder mode')
    action.add_argument(
        '-p', '--player', action='store_true', help='open in player mode')
    action.add_argument(
        '-v', '--visualize', action='store_true', help='open in audio visualize mode (from microphone)')
    action.add_argument(
        '-n', '--netvisualize', action='store_true', help='open in audio visualize mode (from net)')


    # parser.add_argument(
    #     '--ip', default="127.0.0.1", help='IP address to communicate with')
    # parser.add_argument(
    #     '--port', type=int, default=35852, help='port to communicate with')

    # parser.add_argument(
    #     '-d', '--device',
    #     help='input device (numeric ID or substring)')
    # parser.add_argument(
    #     '-b', '--blocksize', type=int, help='block size (in samples)')
    # # If we want, we can remove the default sample rate, and it will default to the device default
    # parser.add_argument(
    #     '-s', '--samplerate', type=float, default=16000, help='sampling rate of audio device')
    # parser.add_argument(
    #     '-t', '--timeout', type=float, help='maximum time to record/play (in seconds)')    
    return parser.parse_args(remaining)

def ready_arguments(args):
    additional_args = {}
    # Add specific sample rate
    if args.samplerate is not None:
        additional_args["samplerate"] = args.samplerate
    # Add specific block size
    if args.blocksize is not None:
        additional_args["blocksize"] = args.blocksize
    # Use specific device
    if args.device is not None:
        additional_args["device"] = args.device
    return additional_args

def ready_address(args):
    return (args.ip, args.port)

def record(args):
    # additional_args = ready_arguments(args)
    # address = ready_address(args)
    # listener = SocketSender(address, args.timeout)
    # receiver = Recorder(listener, additional_args)
    # receiver.listen()

    additional_args = {"samplerate":16000.0, "blocksize":1024}
    address = ("127.0.0.1", 35852)
    listener = SocketSender(address, timeout=3.1) # timeout can be any float, or None for unlimited time
    receiver = Recorder(listener, additional_args)
    receiver.listen()

def record_and_visualize(args):
    additional_args = {"samplerate":16000.0, "blocksize":1024}
    listener = AudioVisualizer(samplerate = additional_args["samplerate"], blocking_time=None)
    receiver = Recorder(listener, additional_args)
    receiver.listen()

def recv_and_visualize(args):
    additional_args = {"samplerate":16000.0, "blocksize":1024}
    listener = AudioVisualizer(samplerate = additional_args["samplerate"], blocking_time=0.01)
    receiver = SocketReceiver(listener, ("127.0.0.1", 35852), blocksize=additional_args["blocksize"])
    receiver.listen()

def main(args):
    if args.recorder:
        print("In recorder mode")
        record(args)
    if args.player:
        print("in player mode")
    if args.visualize:
        record_and_visualize(args)
    if args.netvisualize:
        recv_and_visualize(args)

if __name__ == '__main__':
    args = parse_arguments()
    main(args)
