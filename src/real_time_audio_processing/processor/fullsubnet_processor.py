#!/usr/bin/env python3

"""Processor to reduce noise in an audio stream. See the documentation in the DCCRN project for how it is done.
"""

from __future__ import absolute_import

import importlib

import toml

from src.FullSubNet.audio_zen.acoustics.feature import mag_phase
from src.FullSubNet.audio_zen.inferencer.base_inferencer import BaseInferencer
from .processor import Processor

from ...FullSubNet.fullsubnet.model import Model
from ...DCCRN.utils import remove_pad
from ..utils.raw_samples_converter import raw_samples_to_array, array_to_raw_samples
from ...FullSubNet.audio_zen.inferencer import base_inferencer
import torch

def initialize_module(path: str, args: dict = None, initialize: bool = True):
    """
    Load module dynamically with "args".

    Args:
        path: module path in this project.
        args: parameters that passes to the Class or the Function in the module.
        initialize: initialize the Class or the Function with args.

    Examples:
        Config items are as follows：

            [model]
            path = "model.full_sub_net.FullSubNetModel"
            [model.args]
            n_frames = 32
            ...

        This function will:
            1. Load the "model.full_sub_net" module.
            2. Call "FullSubNetModel" Class (or Function) in "model.full_sub_net" module.
            3. If initialize is True:
                instantiate (or call) the Class (or the Function) and pass the parameters (in "[model.args]") to it.
    """
    module_path = ".".join(path.split(".")[:-1])
    class_or_function_name = path.split(".")[-1]

    #module = importlib.import_module(module_path)
    class_or_function = Model

    if initialize:
        if args:
            return class_or_function(**args)
        else:
            return class_or_function()
    else:
        return class_or_function

def _load_model(model_config, checkpoint_path, device):
    model = initialize_module(model_config["path"], args=model_config["args"], initialize=True)
    model_checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_static_dict = model_checkpoint["model"]
    epoch = model_checkpoint["epoch"]
    print(f"Loaded the torch model tar, epoch： {epoch}.")

    model.load_state_dict(model_static_dict)
    model.to(device)
    model.eval()
    return model, model_checkpoint["epoch"]

class FullsubnetProcessor(Processor):
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
        #        self.model = DCCRN.load_model(model_path)

        configuration = toml.load('C:/Users/wajd_/Desktop/Technion/speech_denoiser_fork/models/inference.toml')
        device = 'cpu' if torch.cuda.is_available() else 'cpu'
        self.ratio_power = ratio_power
        self.model, _ = _load_model(configuration["model"], 'C:/Users/wajd_/Desktop/Technion/speech_denoiser_fork/models/fullsubnet_best_model_58epochs.tar', device)

    def clean_noise(self, samples):
        """Use the DCCRN model and clean the noise from the given samples.

        Args:
            samples (list): List of samples to clean from noise.
        """
        # Pass the audio through the DCCRN model
        noisy_complex = torch.stft((torch.Tensor([samples])), n_fft=512, hop_length=256)

        noisy_mag = (noisy_complex.pow(2.).sum(-1) + 1e-8).pow(0.5 * 1.0).unsqueeze(1)

        estimated_samples = torch.istft(self.model(noisy_mag).detach().permute(0, 2, 3, 1), 512, 256)

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

