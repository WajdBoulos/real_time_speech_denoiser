#!/usr/bin/env python3

"""Special writer that first calls a processor on the data in order to process it, and then calls the actual writer
in order to write the data.
"""

from __future__ import absolute_import

from .writer import Writer


class ProcessorWriter(Writer):
    """Get audio data from a reader, put it through a processor, and send it to another writer"""
    def __init__(self, processor, writer):
        """Initialize a ProcessorWriter object.
        
        Args:
            processor (Processor):  processor to call with the data.
            writer (Writer):        writer to call with the data.
        """
        self.processor = processor
        self.writer = writer

    def data_ready(self, data):
        """Let the processor process the data, then call the writer with the data.

        Args:
            data (buffer):        data to write. It is a buffer with length of blocksize*sizeof(dtype).
        """
        self.processor.process(data)
        self.writer.data_ready(data)

    def wait(self):
        """Wait for the processor and writer to process the data.

        Returns:
            True if one of the processor or writer returns True.
        """
        status = self.processor.wait()
        # status must be after self.writer.wait(), to make sure that even if status is true, self.writer.wait() is still called.
        status = self.writer.wait() or status
        return status

    def finalize(self):
        """Wait for the processor and writer to finish.
        """
        self.processor.finalize()
        self.writer.finalize()