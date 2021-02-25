from __future__ import absolute_import

from .writer import Writer


class ProcessorWriter(Writer):
    """Get audio data from a reader, put it through a processor, and send it to another writer"""
    def __init__(self, processor, writer):
       self.processor = processor
       self.writer = writer

    def data_ready(self, data):
        self.processor.process(data)
        self.writer.data_ready(data)

    def wait(self):
    	return self.writer.wait()