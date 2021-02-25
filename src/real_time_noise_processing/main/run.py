#!/usr/bin/env python3

""" Run a combination of the objects in the library.
"""

from __future__ import absolute_import

from ..reader.socket_reader import SocketReader
from ..reader.microphone_reader import MicrophoneReader

from ..writer.socket_writer import SocketWriter
from ..writer.audio_visualizer import AudioVisualizer
from ..writer.speaker_player import SpeakerPlayer
from ..writer.processor_writer import ProcessorWriter

from ..processor.processor import Processor #TODO: Delete me!

known_readers = {
                    "microphone_reader":MicrophoneReader,
                    "socket_reader":SocketReader,
                }
known_writers = {
                    "audio_visualizer":AudioVisualizer,
                    "socket_writer":SocketWriter,
                    "speaker_player":SpeakerPlayer,
                    "processor_writer":ProcessorWriter,
                }
known_processors = {
                    "processor":Processor, #TODO: Delete me!
                    }

def initialize_objects(object_list):
    # for reader_name, reader_data in object_list["readers"].items():
    #     writer = known_writers[reader_data["writer"]["name"]](**reader_data["writer"]["args"])
    #     reader = known_readers[reader_name](writer, **reader_data["args"])
    #     reader.read()

    # writer = known_writers["speaker_player"](additional_args={"samplerate":16000.0, "blocksize":1024})
    writer1 = known_writers["speaker_player"](additional_args={"samplerate":16000.0, "blocksize":1024})
    processor1 = known_processors["processor"]()
    writer2 = known_writers["processor_writer"](processor=processor1, writer=writer1)
    reader = known_readers["microphone_reader"](writer2, additional_args={"samplerate":16000.0, "blocksize":1024})
    reader.read()

def main():
    # Read the YAML of objects to create
    # Initialize the objects
    
    # Placeholder simple reader and writer to test this file
    classes = {
        # "readers": {
        #     "microphone_reader": {
        #         "args": {"additional_args":{"samplerate":16000.0, "blocksize":1024}},
        #         "writer": {
        #             "name":"speaker_player", #"name":"audio_visualizer",
        #             "args": {"additional_args":{"samplerate":16000.0, "blocksize":1024}},#"args": {"samplerate":16000.0, "blocking_time":None, "duration":1}
        #         }
        #     }
        # }
    }
    initialize_objects(classes)

if __name__ == '__main__':
    main()