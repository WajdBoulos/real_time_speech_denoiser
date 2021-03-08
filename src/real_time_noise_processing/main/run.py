#!/usr/bin/env python3

""" Run a combination of the objects in the library.
"""

from __future__ import absolute_import

from ..reader.socket_reader import SocketReader
from ..reader.microphone_reader import MicrophoneReader
from ..reader.file_reader import FileReader

from ..writer.socket_writer import SocketWriter
from ..writer.audio_visualizer import AudioVisualizer
from ..writer.speaker_player import SpeakerPlayer
from ..writer.processor_writer import ProcessorWriter
from ..writer.file_writer import FileWriter

from ..processor.splitter import Splitter
from ..processor.pipeline import Pipeline
from ..processor.multiplier import Multiplier

import argparse
import yaml

known_readers = {
                    "microphone_reader":MicrophoneReader,
                    "socket_reader":SocketReader,
                    "file_reader":FileReader,
                }
known_writers = {
                    "audio_visualizer":AudioVisualizer,
                    "socket_writer":SocketWriter,
                    "speaker_player":SpeakerPlayer,
                    "processor_writer":ProcessorWriter,
                    "file_writer":FileWriter,
                }
known_processors = {
                    "splitter":Splitter,
                    "pipeline":Pipeline,
                    "multiplier":Multiplier
                    }

def initialize_objects(object_list):
    # Create each of the processors and add them to the pipeline list in order
    pipeline = []
    for processor in object_list["pipeline"]:
        pipeline.append(known_processors[processor["type"]](**processor["args"]))
    # Create each writer and add them to the list in order (the order should not matter)
    writers = []
    for writer in object_list["writers"]:
        writers.append(known_writers[writer["type"]](**writer["args"]))

    if len(writers) > 1:
        # Create a splitter for all the writers
        splitter_processor = known_processors["splitter"](writers[:-1])
        # Add the splitter as the last processor in the pipeline (so all the processors that change the data will run before it)
        pipeline.append(splitter_processor)
        # Set the final writer as the last writer in the list
        final_writer = writers[-1]
    else:
        # Set the only writer as the final writer (there must be at least one writer)
        final_writer = writers[0]
    if len(pipeline) > 0:
        # Create a pipeline
        pipeline_processor = known_processors["pipeline"](pipeline)
        # Replace the final writer with a processor writer that calls the pipeline and then the writer
        final_writer = known_writers["processor_writer"](pipeline_processor, final_writer)

    # Create the reader coupled to the final writer    
    reader = known_readers[object_list["reader"]["type"]](final_writer, **object_list["reader"]["args"])
    reader.read()

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-f', '--filename', help='setting file with classes to create', required=True)
    return parser.parse_args()

def main():
    # Read the YAML of objects to create
    # Initialize the objects
    
    args = parse_arguments()

    with open(args.filename, "r") as f:
        classes = yaml.load(f, yaml.SafeLoader)

    initialize_objects(classes)

if __name__ == '__main__':
    main()