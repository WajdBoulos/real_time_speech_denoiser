#!/usr/bin/env python3

from __future__ import absolute_import

from .processor import Processor
import struct

import torch
from ...DCCRN.DCCRN import DCCRN
from ...DCCRN.utils import remove_pad

class DCCRNProcessor(Processor):
    def __init__(self, model_path, sample_size=4):
        self.sample_size = sample_size
        self.model = DCCRN.load_model(model_path)

        if self.sample_size == 4:
            self.unpack_string = "f"
            self.should_use_int = False
        elif self.sample_size == 2:
            self.unpack_string = "h"
            self.should_use_int = True
        else:
            raise ValueError(f"unsupported sample size {self.sample_size}")

    def process(self, data):
        samples = [struct.unpack(self.unpack_string, data[i:i+self.sample_size])[0] for i in range(0, len(data), self.sample_size)]
        estimated_samples = self.model(torch.Tensor([samples]))
        with torch.no_grad():
            clean_samples = remove_pad(estimated_samples, [len(samples)])

        for i, clean_sample in zip(range(0, len(data), self.sample_size), clean_samples[0]):
            if self.should_use_int:
                clean_sample = int(clean_sample)
            clean_sample_bytes = struct.pack(self.unpack_string, clean_sample)
            for j, value in enumerate(clean_sample_bytes):
                data[i + j:i + j + 1] = bytes([value])


    def wait(self):
        # Never finish
        return False

    def finalize(self):
        # Nothing to do here
        return

