#!/usr/bin/env python3

from __future__ import absolute_import

from .processor import Processor

from ...DCCRN.DCCRN import DCCRN
from ...DCCRN.utils import remove_pad
from ..utils.raw_samples_converter import raw_samples_to_array, array_to_raw_samples

import torch

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

    def clean_noise(self, samples):
        estimated_samples = self.model(torch.Tensor([samples]))
        with torch.no_grad():
            clean_samples = remove_pad(estimated_samples, [len(samples)])
        return clean_samples[0]

    def process(self, data):
        # Get the samples from the data
        samples = raw_samples_to_array(data, self.sample_size)

        if self.should_overlap:
            if self.previous_original is None:
                # Save the last window, zero the current window, and return
                self.previous_original = samples
                self.previous_processed = [0] * len(samples) + list(self.clean_noise(samples))
                clean_samples = [0] * len(samples)
                array_to_raw_samples(clean_samples, data, self.sample_size)
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
        array_to_raw_samples(clean_samples, data, self.sample_size)


    def wait(self):
        # Never finish
        return False

    def finalize(self):
        # Nothing to do here
        return

