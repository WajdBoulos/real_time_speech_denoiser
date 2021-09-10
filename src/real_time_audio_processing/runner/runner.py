#!/usr/bin/env python3

"""Runner object to make the process of creating readers, writers, and processors easier to manage and allow for easy
expanding of this library both inside and in other libraries.
"""

from __future__ import absolute_import

import importlib

# The following are dictionaries of all the readers, writers, and processors that are native to this library. It is
# easily possible to create other readers, writer, and processors, both inside of this library and in the code of a
# user of this library. To create additional readers, writers and processors inside this library, simply add the object
# to the correct list below in the same manner as the other objects that are in the list currently. To create
# additional objects outside of this library, do not add the object to these dictionaries, and instead use the
# functionality that is exported in the Runner object in order to use different objects.
# Each of the dictionaries here is made out of a key for each class (the key that will need to appear in the object
# list in order to create the corresponding class), and its value, which is a lambda function that imports the class
# and returns it.
# The reason I chose to put a function that imports the class and returns it instead of importing all the classes and
# only putting in the dict the class itself, is that the importing of some classes can be quite heavy, and is not
# needed if the object is not one of the objects needed to be created as detailed in the object list.
known_readers = {
                    "microphone_reader": lambda:importlib.import_module("..reader.microphone_reader", __package__).MicrophoneReader,
                    "socket_reader": lambda:importlib.import_module("..reader.socket_reader", __package__).SocketReader,
                    "file_reader": lambda:importlib.import_module("..reader.file_reader", __package__).FileReader,
                    "multi_file_reader": lambda:importlib.import_module("..reader.multi_file_reader", __package__).FileReader,
}
known_writers = {
                    "audio_visualizer": lambda:importlib.import_module("..writer.audio_visualizer", __package__).AudioVisualizer,
                    "socket_writer": lambda:importlib.import_module("..writer.socket_writer", __package__).SocketWriter,
                    "speaker_player": lambda:importlib.import_module("..writer.speaker_player", __package__).SpeakerPlayer,
                    "processor_writer": lambda:importlib.import_module("..writer.processor_writer", __package__).ProcessorWriter,
                    "file_writer": lambda:importlib.import_module("..writer.file_writer", __package__).FileWriter,
                    "multi_file_writer": lambda:importlib.import_module("..writer.multi_file_writer", __package__).FileWriter,
}
known_processors = {
                    "splitter": lambda:importlib.import_module("..processor.splitter", __package__).Splitter,
                    "pipeline": lambda:importlib.import_module("..processor.pipeline", __package__).Pipeline,
                    "multiplier": lambda:importlib.import_module("..processor.multiplier", __package__).Multiplier,
                    "DCCRN_processor": lambda:importlib.import_module("..processor.dccrn_processor", __package__).DCCRNProcessor,
                    "Fullsubnet_processor": lambda:importlib.import_module("..processor.fullsubnet_processor", __package__).FullsubnetProcessor,
                    }

class Runner(object):
    """Runner object to create and run the objects for the audio processing"""
    def __init__(self, object_list):
        """
        Args:
            object_list (dictionary):   Dictionary containing any of the keys "reader", "pipeline", "writers".
                It is possible to call this function with or without any one of those keys (even an empty dict is
                allowed). In such a case, calls to add_writers and add_reader should be made to add the needed objects
                for running.
       
        """
        self.reader = None
        self.processors = []
        self.writers = []

        # Create and add any processors that appeared in the object list
        if "pipeline" in object_list:
            self.add_known_processors(object_list["pipeline"])
        # Create and add any writers that appeared in the object list
        if "writers" in object_list:
            self.add_known_writers(object_list["writers"])

        # Remember if a reader appears in the object list, in order to create it later
        self.known_reader = None
        if "reader" in object_list:
            self.known_reader = object_list["reader"]

    def add_known_processors(self, processor_list):
        """Create known processors, and add them to the end of the pipeline of processors.

        Args:
            processor_list (list):  List of processors to create and add to the pipeline. Each item in this list should
                be a dictionary with the key "type" and a type of known processor, and a key "args" and the arguments
                to pass at the creation of this processor.
        """
        # Create each of the processors and add them to the pipeline list in order
        for processor in processor_list:
            self.processors.append(known_processors[processor["type"]]()(**processor["args"]))

    def add_known_writers(self, writer_list):
        """Create known writers, and add them to the end of the list of writers.

        Args:
            writer_list (list):  List of writers to create and add to the list. Each item in this list should
                be a dictionary with the key "type" and a type of known writer, and a key "args" and the arguments
                to pass at the creation of this writer.
        """
        # Create each writer and add them to the list in order (the order should not matter)
        for writer in writer_list:
            self.writers.append(known_writers[writer["type"]]()(**writer["args"]))

    def add_known_reader(self, reader):
        """Create a known reader as the main reader of this runner.
        After a call to this method, the user must not call any other method of this runner object. Any writer or
        processor the user wishes to add must be added before calling this function.

        Args:
            reader (dict):  Dictionary with "type" and the type of a known reader, and "args" and the arguments to pass
                at the creation of this reader.
        """
        # Create the reader coupled to the final writer
        self.reader = known_readers[reader["type"]]()(self.get_writer(), **reader["args"])
        
    def add_processors(self, processor_list):
        """Add instances of processors to the end of the pipeline of processors.

        Args:
            processor_list (list):  List of actual, initialized processors to add to the pipeline. Each item in this
                list should be a processor, that inherited and implements a Processor object.
        """
        self.processors.extend(processor_list)

    def add_writers(self, writer_list):
        """Add instances of writers to the writers to use in this runner.

        Args:
            writer_list (list):  List of actual, initialized writers to add to the writers. Each item in this
                list should be a writer, that inherited and implements a Writer object.
        """
        self.writers.extend(writer_list)

    def add_reader(self, reader):
        """Add a reader to the runner.

        Using this method will override any other reader, writer, or processor that the user added to this runner
        object. If the user wishes to use the writers and processors already added to this object as part of their
        reader, see the get_writer method description.
        After a call to this method, the user must not call any other method of this runner object. Any writer or
        processor the user wishes to add must be added before calling this function.

        Args:
            reader (Reader):  Actual, initialized reader to add as the reader of this runner. This should be a reader,
                that inherited and implements a Reader object.
        """
        self.reader = reader

    def get_writer(self):
        """Get the writer that is to be used with a reader in this runner.
        The writer returned from this method is the final writer, which (if needed) already includes any pipeline or
        splitter writers.
        This method is intended as an easy way for the user to create their own reader and put it as the main reader
        of the runner. To do this, the user can call get_writer, use this writer in the creation of a reader, call
        add_reader with the new reader, and then use this runner normally.
        After a call to this function, the user must call only the methods add_reader or add_known_reader, before calling run.

        Returns:
            The final writer to be used by this runner.
        """
        # Make sure that there is at least one writer
        if len(self.writers) == 0:
            raise ValueError("There must be at least one writer in a runner")
        # Check if we only have one writer
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

        # Check if there are processors, and if there are, create a pipeline for them
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
        """Run the reader of this runner, which in turn will run all the writers and processors that are in it.
        This function blocks and only returns once the reader finishes its operation.
        After calling this function, this runner is useless, and should be discarded, as the user must not call
        any other function of this runner after this function ends.
        """
        # Check if there is already an established reader, or add the known reader as the reader of this runner
        if self.reader is None:
            # Make sure that there is even a known reader to creates
            if self.known_reader is None:
                raise ValueError("No reader was added to this runner")
            # Create the known reader
            self.add_known_reader(self.known_reader)

        # Start the runner
        self.reader.read()
        # Make sure the user never calls this function again
        self.reader = None
