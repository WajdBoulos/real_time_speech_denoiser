#!/usr/bin/env python
# Created on 2018/12
# Author: Kaituo XU

import argparse
import json
import os

import librosa
from tqdm import tqdm

def preprocess_one_dir(data_dir, json_dir, json_filename, sample_rate=8000):
    file_infos = []
    data_dir = os.path.abspath(data_dir)
    wav_list = os.listdir(data_dir)
    print('Current list is:' + str(data_dir))
    for wav_file in tqdm(wav_list):
        if not wav_file.endswith('.wav'):
            continue
        wav_path = os.path.join(data_dir, wav_file)
        samples, _ = librosa.load(wav_path, sr=sample_rate)
        file_infos.append((wav_path, len(samples)))
    if not os.path.exists(json_dir):
        os.makedirs(json_dir)
    with open(os.path.join(json_dir, json_filename + '.json'), 'w') as f:
        json.dump(file_infos, f, indent=4)


def preprocess(data_dir, json_dir, sample_rate):
    for data_type in ['tr',]:
        for speaker in ['mix', 'clean', 'noise']:
            preprocess_one_dir(os.path.join(data_dir, data_type, speaker),
                               os.path.join(json_dir, data_type),
                               speaker,
                               sample_rate=sample_rate)


if __name__ == "__main__":

    data_dir = "../egs/SE_dataset"
    json_dir = "../egs/SE_dataset"
    sample_rate = 16000
    preprocess(data_dir, json_dir, sample_rate)
