#!/usr/bin/env python3

"""Processor to reduce noise in an audio stream. See the documentation in the DCCRN project for how it is done.
"""

from __future__ import absolute_import

from src.FullSubNet.audio_zen.acoustics.feature import mag_phase
from .processor import Processor

from ...FullSubNet.inferencer import Inferencer
from ...FullSubNet.fullsubnet.model import Model
from ...DCCRN.utils import remove_pad
from ..utils.raw_samples_converter import raw_samples_to_array, array_to_raw_samples
import torch
import cProfile

def decompress_cIRM(mask, K=10, limit=9.9):
    mask = limit * (mask >= limit) - limit * (mask <= -limit) + mask * (torch.abs(mask) < limit)
    mask = -K * torch.log((K - mask) / (K + mask))
    return mask

class FullsubnetProcessor(Processor):
    """Reduce noise in the audio.
    """
    def __init__(self, model_path, should_overlap=False, ratio_power=1, sample_size=4):
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
        #        self.model = DCCRN.load_model(model_path)

        self.ratio_power = ratio_power
        self.model = Model(
            sb_num_neighbors=15,
            fb_num_neighbors=0,
            num_freqs=257,
            look_ahead=2,
            sequence_model="LSTM",
            fb_output_activate_function="ReLU",
            sb_output_activate_function=None,
            fb_model_hidden_size=512,
            sb_model_hidden_size=384,
            weight_init=False,
            norm_type="offline_laplace_norm",
            num_groups_in_drop_band=2,
        )

        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint['model'])
        self.should_overlap = should_overlap
        if self.should_overlap:
            self.previous_original = None
        self.ratio_power = ratio_power

    def clean_noise(self, samples):
        """Use the DCCRN model and clean the noise from the given samples.

        Args:
            samples (list): List of samples to clean from noise.
        """
        # Pass the audio through the DCCRN model
        #clean = Inferencer.full_band_crm_mask(samples)
        noisy_complex = torch.stft((torch.Tensor([samples])), n_fft=512, hop_length=256, window=torch.hann_window(512), return_complex=True)

        noisy_mag = torch.abs(noisy_complex).unsqueeze(1)
        # pr = cProfile.Profile()
        # pr.enable()

        pred_crm = self.model(noisy_mag).detach().permute(0, 2, 3, 1)


        # pr.disable()
        # pr.print_stats(sort='time')

        pred_crm = decompress_cIRM(pred_crm)
        enhanced_real = pred_crm[..., 0] * noisy_complex.real - pred_crm[..., 1] * noisy_complex.imag
        enhanced_imag = pred_crm[..., 1] * noisy_complex.real + pred_crm[..., 0] * noisy_complex.imag
        enhanced_complex = torch.stack((enhanced_real, enhanced_imag), dim=-1)

        estimated_samples = torch.istft(enhanced_complex, 512, 256)

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
        """Always return False, to never finish.
        """
        return False

    def finalize(self):
        """Do nothing, as all the processing was done in the process function.
        """
        return

