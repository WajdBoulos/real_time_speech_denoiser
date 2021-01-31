#!/usr/bin/env python3

""" Read audio data from a socket.
"""

from __future__ import absolute_import

from .reader import Reader
import socket

class SocketReader(Reader):
    """Read audio from a socket and send each block of samples to a writer"""
    def __init__(self, writer, address, blocksize, sample_size=4):
        """Initialize a SocketReader object.
        This reader opens a listening socket, and tries to get data from the first socket that connects to it.

        This reader works in the main thread only, so the writer can not block for too long when writer.wait is called.

        Args:
            writer (Writer):        Writer object to give the data to.
            address (tuple):        Local address to bind the listening port to. A tuple of (address, port).
            blocksize (int):        Block size to get from the socket. The writer will only be called after this amount
                of samples arrived from the socket.
            sample_size (int):      Amount of bytes in each sample.

        """
        self.writer = writer
        self.address = address
        self.blocksize = blocksize
        self.sample_size = sample_size
        self.initialize_socket()

    def initialize_socket(self):
        """Initialize a listening socket, and wait for a socket writer to connect to it.
        After this function returns, self.socket should be a socket connected to a socket writer.
        """
        self.listening_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listening_socket.bind((self.address))
        self.listening_socket.listen(1)
        self.socket, remote_addr = self.listening_socket.accept()
        print("got connection from", remote_addr)
        self.listening_socket.close()

    def read(self):
        """Read from the connected socket, and send it to a writer.

        This function works in the main thread only. 
        This means that the writer can NOT block when writer.wait is called.
        """
        # Wait for the writer to tell us we need to stop reading, or for the socket to close
        # (indicating there is no more data, so there is no reason to continue to call the writer).
        while not self.writer.wait() and self.socket is not None:
            # Read the data from the socket in chunks until we have all the data.
            current_data = []
            remaining_len = self.blocksize * self.sample_size
            while remaining_len != 0 and self.socket is not None:
                current_data.append(self.socket.recv(remaining_len))
                remaining_len -= len(current_data[-1])
                if len(current_data[-1]) == 0:
                    self.socket.close()
                    self.socket = None

            # Join the data to one buffer and send it to the writer.
            total_data = b"".join(current_data)
            self.writer.data_ready(total_data)

        if self.socket is not None:
            self.socket.close()
            self.socket = None
