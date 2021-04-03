#!/usr/bin/env python3

""" Run a combination of the objects in the library.
"""

from __future__ import absolute_import

import importlib
import argparse
import yaml

known_readers = {
                    "microphone_reader": lambda:importlib.import_module("..reader.microphone_reader", __package__).MicrophoneReader,
                    "socket_reader": lambda:importlib.import_module("..reader.socket_reader", __package__).SocketReader,
                    "file_reader": lambda:importlib.import_module("..reader.file_reader", __package__).FileReader,
                }
known_writers = {
                    "audio_visualizer": lambda:importlib.import_module("..writer.audio_visualizer", __package__).AudioVisualizer,
                    "socket_writer": lambda:importlib.import_module("..writer.socket_writer", __package__).SocketWriter,
                    "speaker_player": lambda:importlib.import_module("..writer.speaker_player", __package__).SpeakerPlayer,
                    "processor_writer": lambda:importlib.import_module("..writer.processor_writer", __package__).ProcessorWriter,
                    "file_writer": lambda:importlib.import_module("..writer.file_writer", __package__).FileWriter,
                }
known_processors = {
                    "splitter": lambda:importlib.import_module("..processor.splitter", __package__).Splitter,
                    "pipeline": lambda:importlib.import_module("..processor.pipeline", __package__).Pipeline,
                    "multiplier": lambda:importlib.import_module("..processor.multiplier", __package__).Multiplier,
                    "DCCRN_processor": lambda:importlib.import_module("..processor.dccrn_processor", __package__).DCCRNProcessor,
                    }

def initialize_objects(object_list):
    # Create each of the processors and add them to the pipeline list in order
    pipeline = []
    for processor in object_list["pipeline"]:
        pipeline.append(known_processors[processor["type"]]()(**processor["args"]))
    # Create each writer and add them to the list in order (the order should not matter)
    writers = []
    for writer in object_list["writers"]:
        writers.append(known_writers[writer["type"]]()(**writer["args"]))

    if len(writers) > 1:
        # Create a splitter for all the writers
        splitter_processor = known_processors["splitter"]()(writers[:-1])
        # Add the splitter as the last processor in the pipeline (so all the processors that change the data will run before it)
        pipeline.append(splitter_processor)
        # Set the final writer as the last writer in the list
        final_writer = writers[-1]
    else:
        # Set the only writer as the final writer (there must be at least one writer)
        final_writer = writers[0]
    if len(pipeline) > 0:
        # Create a pipeline
        pipeline_processor = known_processors["pipeline"]()(pipeline)
        # Replace the final writer with a processor writer that calls the pipeline and then the writer
        final_writer = known_writers["processor_writer"]()(pipeline_processor, final_writer)

    # Create the reader coupled to the final writer    
    reader = known_readers[object_list["reader"]["type"]]()(final_writer, **object_list["reader"]["args"])
    reader.read()

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-f', '--filename', help='setting file with classes to create', required=True)
    return parser.parse_args()

def load_classes_dict(classes_file_name):
    with open(classes_file_name, "r") as f:
        classes = yaml.load(f, yaml.SafeLoader)
    return classes

def main():
    # Read the YAML of objects to create
    # Initialize the objects
    
    args = parse_arguments()

    classes = load_classes_dict(args.filename)

    initialize_objects(classes)

if __name__ == '__main__':
    main()