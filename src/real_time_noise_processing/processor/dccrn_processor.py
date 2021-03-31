#!/usr/bin/env python3

from __future__ import absolute_import

from .processor import Processor
import struct

import torch
from ...DCCRN.DCCRN import DCCRN
from ...DCCRN.utils import remove_pad

class DCCRNProcessor(Processor):
    def __init__(self, model_path, should_overlap=True, ratio_power=1, sample_size=4):
        """
        note: If should_overlap is True, the first chunk of data will be zeroed, the last chunk of data will be lost,
        and there will be a delay of one chunk of data between the input of this processor and the output.
        """
        self.sample_size = sample_size
        self.model = DCCRN.load_model(model_path)
        self.should_overlap = should_overlap
        if self.should_overlap:
            self.previous_original = None
        self.ratio_power = ratio_power

        if self.sample_size == 4:
            self.unpack_string = "f"
            self.should_use_int = False
        elif self.sample_size == 2:
            self.unpack_string = "h"
            self.should_use_int = True
        else:
            raise ValueError(f"unsupported sample size {self.sample_size}")

    def clean_noise(self, samples):
        estimated_samples = self.model(torch.Tensor([samples]))
        with torch.no_grad():
            clean_samples = remove_pad(estimated_samples, [len(samples)])
        return clean_samples[0]

    def process(self, data):
        # Get the samples from the data
        samples = [struct.unpack(self.unpack_string, data[i:i+self.sample_size])[0] for i in range(0, len(data), self.sample_size)]

        def convert_back_samples(clean_samples):
            for i, clean_sample in zip(range(0, len(data), self.sample_size), clean_samples):
                if self.should_use_int:
                    clean_sample = int(clean_sample)
                clean_sample_bytes = struct.pack(self.unpack_string, clean_sample)
                for j, value in enumerate(clean_sample_bytes):
                    data[i + j:i + j + 1] = bytes([value])

        if self.should_overlap:
            if self.previous_original is None:
                # Save the last window, zero the current window, and return
                self.previous_original = samples
                self.previous_processed = [0] * len(samples) + list(self.clean_noise(samples))
                clean_samples = [0] * len(samples)
                convert_back_samples(clean_samples)
                return
            # Process the current samples
            current_processed = self.clean_noise(self.previous_original + samples)
            # Generate the output vector by combining the end of the last window and the start of the current window
            combined_samples = []
            for i, (previous_sample, current_sample) in enumerate(zip(self.previous_processed[len(samples):], current_processed[:len(samples)])):
                ratio = ((i + 1) / len(samples))
                ratio = ratio ** self.ratio_power
                combined_samples.append(ratio * current_sample + (1 - ratio) * previous_sample)
            clean_samples = combined_samples
            # Save the last samples for the next time
            self.previous_original = samples
            self.previous_processed = current_processed
        else:
            # Estimate the clean samples using the model
            clean_samples = self.clean_noise(samples)

        # Convert the samples back to data
        convert_back_samples(clean_samples)


    def wait(self):
        # Never finish
        return False

    def finalize(self):
        # Nothing to do here
        return

