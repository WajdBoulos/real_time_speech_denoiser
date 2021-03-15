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
        status = self.processor.wait()
        # status must be after self.writer.wait(), to make sure that even if status is true, self.writer.wait() is still called.
        status = self.writer.wait() or status
        return status

    def finalize(self):
        self.processor.finalize()
        self.writer.finalize()