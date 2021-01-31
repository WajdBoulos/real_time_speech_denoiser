#!/usr/bin/env python3

""" Write audio data to a socket.
"""

from __future__ import absolute_import

from .writer import Writer
import socket
import select

class SocketWriter(Writer):
    """Get audio data from a reader and send it to a socket"""
    def __init__(self, dest, timeout=None):
        """Initialize a SocketWriter object.
        This writer connects to a remote socket, and sends it every block of data it gets.

        The wait function of this object always blocks, until either the timeout passes or the other end of the socket
            closes, and in any of those cases the socket is closed and the wait function returns True.
        This behavior should be changed in the future, to allow running the wait function in non-blocking mode.

        Args:
            dest (tuple):           Remote address to connect the socket to. A tuple of (address, port).
            timeout (float):        Maximum time (in seconds) to keep sending data through this socket. After this
            amount of time, the socket will be closed and the wait function will return True. If None is used, no
            timeout will be defined for the socket and it will continue to send data until the other end of the
            socket closes.

        """
        self.dest = dest
        self.timeout = timeout
        self.initialize_socket()

    def initialize_socket(self):
        """Connect to the remote socket.
        After this function returns, self.socket should be a socket connected to a socket reader.
        """
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.dest))

    def data_ready(self, data):
        """Send data to the remote socket.

        Args:
            data (buffer):        data to write. It is a buffer with length of blocksize*sizeof(dtype).
        """
        # Loop while there is data not sent
        while data:
            if self.socket is None:
                break
            # Send the data
            sent_len = self.socket.send(data)
            # Set data to only be any leftover we did not send
            data = data[sent_len:]
            # If we did not send any data, assume there was an error and close the socket
            if sent_len == 0:
                print("sent 0 bytes")
                self.socket.close()
                self.socket = None

    def wait(self):
        """Wait for the remote socket to close or the timeout to pass.

        This function blocks, so this writer should not be used when the reader runs in the main thread only.
        Returns:
            Always True.
        """
        # Because no data should be received in this socket,
        # this actually just waits for the other end of the socket to close
        if self.timeout is not None:
            # Wait for the socket to close or a maximum time
            select.select([self.socket], [], [], self.timeout)
        else:
            # Wait for the socket to close
            select.select([self.socket], [], [])
        self.socket.close()
        self.socket = None
        print("socket was ready to read")
        return True
