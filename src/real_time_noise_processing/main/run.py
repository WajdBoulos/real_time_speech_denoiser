#!/usr/bin/env python3

""" Run a combination of the objects in the library.
"""

from __future__ import absolute_import

from ..reader.socket_reader import SocketReader
from ..reader.microphone_reader import MicrophoneReader
from ..writer.socket_writer import SocketWriter
from ..writer.audio_visualizer import AudioVisualizer

known_readers = {
                    "microphone_reader":MicrophoneReader,
                    "socket_reader":SocketReader,
                }
known_writers = {
                    "audio_visualizer":AudioVisualizer,
                    "socket_writer":SocketWriter,
                }
known_processors = {
                    }

def initialize_objects(object_list):
    for reader_name, reader_data in object_list["readers"].items():
        writer = known_writers[reader_data["writer"]["name"]](**reader_data["writer"]["args"])
        reader = known_readers[reader_name](writer, **reader_data["args"])
        reader.read()

def main():
    # Read the YAML of objects to create
    # Initialize the objects
    
    # Placeholder simple reader and writer to test this file
    classes = {
        "readers": {
            "microphone_reader": {
                "args": {"additional_args":{"samplerate":16000.0, "blocksize":1024}},
                "writer": {
                    "name":"audio_visualizer",
                    "args": {"samplerate":16000.0, "blocking_time":None}
                }
            }
        }
    }
    initialize_objects(classes)

if __name__ == '__main__':
    main()