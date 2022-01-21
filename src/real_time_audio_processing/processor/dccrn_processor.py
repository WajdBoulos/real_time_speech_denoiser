#!/usr/bin/env python3

"""Processor to reduce noise in an audio stream. See the documentation in the DCCRN project for how it is done.
"""

from __future__ import absolute_import
import scipy

from .processor import Processor

from ...DCCRN.DCCRN import DCCRN
from ...DCCRN.utils import remove_pad
from ..utils.raw_samples_converter import raw_samples_to_array, array_to_raw_samples

import torch

class DCCRNProcessor(Processor):
    """Reduce noise in the audio.
    """
    def __init__(self, model_path, should_overlap=True, ratio_power=1, sample_size=4):
        """Initialize a Multiplier processor.
        
        Args:
            model_path (str):       Path to the model to use in the DCCRN NN.
            should_overlap (bool):  Should the processor be run in a delay of one block, in order to overlap each block
                half with the next block and half with the previous block, to reduce artifacts that can cause the noise
                reduction to work in a worse manner when working with small blocks of audio.
                If should_overlap is True, the first chunk of data will be zeroed, the last chunk of data will be lost,
                and there will be a delay of one chunk of data between the input of this processor and the output.
            ratio_power (int):      Ratio for how fast to transfer from one block to the next. Only used when
                should_overlap is set to True. Higher numbers mean that the last window will fade faster.
            sample_size (int):      Size of each sample of raw bytes. Used in the conversion from the raw bytes to the
                actual sample value.
        """
        self.sample_size = sample_size
        # Ready the NN model used by DCCRN
        self.model_path = model_path
        self.model = DCCRN.load_model(model_path)
        self.should_overlap = should_overlap
        if self.should_overlap:
            self.previous_original = None
        self.ratio_power = ratio_power

    def reset(self):
        self.model = DCCRN.load_model(self.model_path)
        self.previous_original = None
        self.previous_processed = None

    def clean_noise(self, samples):
        """Use the DCCRN model and clean the noise from the given samples.

        Args:
            samples (list): List of samples to clean from noise.
        """
        # Pass the audio through the DCCRN model
        estimated_samples = self.model(torch.Tensor([samples]))
        # Remove padding caused by the model
        with torch.no_grad():
            clean_samples = remove_pad(estimated_samples, [len(samples)])

        # Return a list of clean samples
        return clean_samples[0]

    def process(self, data):
        """Clean the data.

        Args:
            data (buffer):        data to clean. It is a buffer with length of blocksize*sizeof(dtype).
        """
        # Convert the raw data to a list of samples
        samples = raw_samples_to_array(data, self.sample_size)

        if False:
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
        scipy.io.wavfile.write("filename.wav", 16000, clean_samples)



    def wait(self):
        """Always return False, to never finish.
        """
        return False

    def finalize(self):
        """Do nothing, as all the processing was done in the process function.
        """
        self.reset()
        return

