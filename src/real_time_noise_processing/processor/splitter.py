#!/usr/bin/env python3

""" Process audio data.
"""
from __future__ import absolute_import

from .processor import Processor

class Splitter(Processor):
    def __init__(self, writers):
        self.writers = writers

    def process(self, data):
        # Tell all the processors about the data
        for writer in self.writers:
            writer.data_ready(data)

    def wait(self):
        status = False
        # Give the writers time to process the data
        for writer in self.writers:
            status = writer.wait() or status
        return status
