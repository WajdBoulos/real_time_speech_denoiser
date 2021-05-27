#!/usr/bin/env python

# Created on 2018/12
# Author: Kaituo XU

import argparse
import os

import librosa
import torch

from .data import EvalDataLoader, EvalDataset
from .utils import remove_pad
from src.DeepComplexCRN.DCCRN import DCCRN
from tqdm import tqdm


def separate(model_path, mix_dir, mix_json, out_dir, use_cuda, sample_rate, batch_size):
    if mix_dir is None and mix_json is None:
        print("Must provide mix_dir or mix_json! When providing mix_dir, "
              "mix_json is ignored.")

    # Load model

    model = DCCRN.load_model(model_path)
    num_params = 0
    for paramter in model.decoder.parameters():
        num_params += torch.numel(paramter)
    print(num_params)
    model.eval()
    if use_cuda:
        model.cuda()

    # Load data
    eval_dataset = EvalDataset(mix_dir, mix_json,
                               batch_size=batch_size,
                               sample_rate=sample_rate)
    eval_loader = EvalDataLoader(eval_dataset, batch_size=1)
    os.makedirs(out_dir, exist_ok=True)

    def write(inputs, filename, sr=sample_rate):
        librosa.output.write_wav(filename, inputs, sr, norm=True)

    with torch.no_grad():
        for (i, data) in enumerate(eval_loader):
            # Get batch data
            mixture, mix_lengths, filenames = data
            if use_cuda:
                mixture, mix_lengths = mixture.cuda(), mix_lengths.cuda()
            # Forward
            estimate_source = model(mixture)  # [B, C, T]
            # Remove padding and flat
            flat_estimate = remove_pad(estimate_source, mix_lengths)
            # mixture = remove_pad(mixture, mix_lengths)
            # Write result
            for i in tqdm(range(len(filenames))):
                filename = filenames[i]
                filename = os.path.join(out_dir,
                                        os.path.basename(filename).strip('.wav'))
                # write(mixture[i], filename + '.wav') # No need to write noisy audio
                clean = flat_estimate[i]
                write(clean, filename + '.wav')


if __name__ == '__main__':
    #model_path = "../egs/models/DCCRN_sr_16k_batch_16_correct_BN_stft_lookahead.pth"
    model_path = "../egs/models/DCCRN_sr_16k_batch_16_BN.pth"
    mix_dir = "/media/hadaso/hadas-win/DNS-Challenge/DNS-Challenge/datasets/cv/mix"
    mix_json = ""
    out_dir = "/media/hadaso/hadas-win/DNS-Challenge/DNS-Challenge/datasets/cv/denoised"
    use_cuda = 1
    sample_rate = 16000
    batch_size = 1

    separate(model_path, mix_dir, mix_json, out_dir, use_cuda, sample_rate, batch_size)

