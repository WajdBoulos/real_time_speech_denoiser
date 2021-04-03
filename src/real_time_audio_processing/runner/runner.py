#!/usr/bin/env python3

""" Run a combination of the objects in the library.
"""

from __future__ import absolute_import

import importlib

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

class Runner(object):
    """Runner object to create and run the objects for the audio processing"""
    def __init__(self, object_list):
        """
        The object_list is a dictionary containing any of the keys "reader", "pipeline", "writers". 
        It is possible to call this function without any one of those keys (even an empty dict is allowed). In such a
        case, calls to add_writers and add_reader should be made to add the needed objects for running.
        """
        self.reader = None
        self.processors = []
        self.writers = []

        if "pipeline" in object_list:
            self.add_known_processors(object_list["pipeline"])
        if "writers" in object_list:
            self.add_known_writers(object_list["writers"])

        self.known_reader = None
        if "reader" in object_list:
            self.known_reader = object_list["reader"]

    def add_known_processors(self, processor_list):
        # Create each of the processors and add them to the pipeline list in order
        for processor in processor_list:
            self.processors.append(known_processors[processor["type"]]()(**processor["args"]))

    def add_known_writers(self, writer_list):
        # Create each writer and add them to the list in order (the order should not matter)
        for writer in writer_list:
            self.writers.append(known_writers[writer["type"]]()(**writer["args"]))

    def add_known_reader(self, reader):
        """
        After a call to this method, the user must not call any other method of this runner object.
        """
        # Create the reader coupled to the final writer
        self.reader = known_readers[reader["type"]]()(self.get_writer(), **reader["args"])
        
    def add_processors(self, processor_list):
        self.processors.extend(processor_list)

    def add_writers(self, writer_list):
        self.writers.extend(writer_list)

    def add_reader(self, reader):
        """Function to add a reader to the runner. Using this method will override any other reader, writer, or
        processor that the user added to this runner object. If the user wishes to use the writers and processors
        already added to this object as part of their reader, see the get_writer method description.
        After a call to this method, the user must not call any other method of this runner object.
        """
        self.reader = reader

    def get_writer(self):
        """Method to get the writer that is to be used with a reader in this runner.
        The writer returned from this method is the final writer, which (if needed) already includes any pipeline or
        splitter writers.
        This method is intended as an easy way for the user to create their own reader and put it as the main reader
        of the runner. To do this, the user can call get_writer, use this writer in the creation of a reader, call
        add_reader with the new reader, and then use this runner normally.
        After a call to this function, the user must call only the methods add_reader or add_known_reader, before calling run.
        """
        if len(self.writers) == 0:
            raise ValueError("There must be at least one writer in a runner")
        elif len(self.writers) == 1:
            # Set the only writer as the final writer
            final_writer = self.writers[0]
        else:
            # Create a splitter for all the writers
            splitter_processor = known_processors["splitter"]()(self.writers[:-1])
            # Add the splitter as the last processor in the pipeline (so all the processors that change the data will run before it)
            self.processors.append(splitter_processor)
            # Set the final writer as the last writer in the list
            final_writer = self.writers[-1]

        if len(self.processors) > 0:
            # Create a pipeline
            pipeline_processor = known_processors["pipeline"]()(self.processors)
            # Replace the final writer with a processor writer that calls the pipeline and then the writer
            final_writer = known_writers["processor_writer"]()(pipeline_processor, final_writer)

        # Make sure no further calls to add_processors and add_writers are made
        self.processors = None
        self.writers = None

        # Return the writer
        return final_writer

    def run(self):
        # Check if there is already an established reader, or add the known reader as the reader of this runner
        if self.reader is None:
            if self.known_reader is None:
                raise ValueError("No reader was added to this runner")
            self.add_known_reader(self.known_reader)
        self.reader.read()
