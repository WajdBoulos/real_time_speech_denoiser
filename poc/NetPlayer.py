#!/usr/bin/env python3

""" POC for capturing audio in one process and immediately playing it on another.
"""

import sounddevice as sd
import argparse
        
class Recorder(object):
    """Record audio from the microphone and send each block of samples to a listener"""
    def __init__(self, listener, additional_args):
        self.listener = listener
        self.additional_args = additional_args

class Listener(object):
    """Abstract listener class to listen for samples"""
    def get_data(self, data):
        pass

class SocketListener(Listener):
    """Listen to samples from a recorder and send them over socket"""
    def __init__(self, dest):
        self.dest = dest
        self.initialize_socket()

    def initialize_socket(self):
        pass

    def get_data(self, data):
        self.socket.send(data)


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

    parser.add_argument(
        '--ip', default="127.0.0.1", help='IP address to communicate with')
    parser.add_argument(
        '--port', type=int, default=35852, help='port to communicate with')

    parser.add_argument(
        '-d', '--device',
        help='input device (numeric ID or substring)')
    parser.add_argument(
        '-b', '--blocksize', type=int, help='block size (in samples)')
    parser.add_argument(
        '-s', '--samplerate', type=float, default=16000, help='sampling rate of audio device')

    
    args = parser.parse_args(remaining)

def main(args):
    pass

if __name__ == '__main__':
    args = parse_arguments()
    main(args)