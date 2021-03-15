#!/usr/bin/env python3

from __future__ import absolute_import

from .processor import Processor

class Pipeline(Processor):
    def __init__(self, processors):
        self.processors = processors

    def process(self, data):
        # Let each of the processors process the data one after the other
        for processor in self.processors:
            processor.process(data)

    def wait(self):
        status = False
        # Give the processors time to process the data
        for processor in self.processors:
            status = processor.wait() or status
        return status

    def finalize(self):
        for processor in self.processors:
            processor.finalize()
