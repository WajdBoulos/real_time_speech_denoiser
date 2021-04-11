#!/usr/bin/env python3

"""Processor to reduce noise in an audio stream. See the documentation in the DCCRN project for how it is done.
"""

from __future__ import absolute_import

from .processor import Processor

from ...DCCRN.DCCRN import DCCRN
from ...DCCRN.utils import remove_pad
import numpy as np

import torch

class DCCRNProcessor(Processor):
    """Reduce noise in the audio.
    """
    def __init__(self, model_path, should_overlap=True):
        """Initialize a Multiplier processor.
        
        Args:
            model_path (str):       Path to the model to use in the DCCRN NN.
            should_overlap (bool):  Should the processor be run in a delay of one block, in order to overlap each block
                half with the next block and half with the previous block, to reduce artifacts that can cause the noise
                reduction to work in a worse manner when working with small blocks of audio.
                If should_overlap is True, the first chunk of data will be zeroed, the last chunk of data will be lost,
                and there will be a delay of one chunk of data between the input of this processor and the output.
        """
        # Ready the NN model used by DCCRN
        self.model = DCCRN.load_model(model_path)
        
        self.should_overlap = should_overlap
        if self.should_overlap:
            self.previous_original = None

        self.ascending_window = None
        self.descending_window = None

    def clean_noise(self, samples):
        """Use the DCCRN model and clean the noise from the given samples.

        Args:
            samples (list): List of samples to clean from noise.
        """
        # Pass the audio through the DCCRN model
        estimated_samples = self.model(torch.from_numpy(samples.reshape((1, len(samples)))))
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
        samples = data

        if self.should_overlap:
            if self.previous_original is None:
                # Save the last window, zero the current window, and return
                self.previous_original = samples.copy()
                self.previous_processed = np.concatenate((np.zeros(len(samples)), self.clean_noise(samples)))
                data[:] = np.zeros(len(samples))
                return

            # Process the current samples
            current_processed = self.clean_noise(np.concatenate((self.previous_original, samples)))

            # Generate ascending and descending windows with the correct length
            if self.ascending_window is None:
                # We can change these windows to any two windows that sum up to 1 if we want different fading (for example non linear)
                self.ascending_window = np.linspace(0,1,num=len(samples))
                self.descending_window = np.linspace(1,0,num=len(samples))

            # Generate the output vector by combining the end of the last window and the start of the current window
            clean_samples = (self.descending_window * self.previous_processed[len(samples):]) + (self.ascending_window * current_processed[:len(samples)])

            # Save the last samples for the next time
            self.previous_original[:] = samples[:]
            self.previous_processed = current_processed
        else:
            # Estimate the clean samples using the model
            clean_samples = self.clean_noise(samples)

        # Change the output data
        data[:] = clean_samples


    def wait(self):
        """Always return False, to never finish.
        """
        return False

    def finalize(self):
        """Do nothing, as all the processing was done in the process function.
        """
        return

