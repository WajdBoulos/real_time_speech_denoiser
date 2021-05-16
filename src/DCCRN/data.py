# Created on 2018/12
# Author: Kaituo XU
"""
Logic:
1. AudioDataLoader generate a minibatch from AudioDataset, the size of this
   minibatch is AudioDataLoader's batchsize. For now, we always set
   AudioDataLoader's batchsize as 1. The real minibatch size we care about is
   set in AudioDataset's __init__(...). So actually, we generate the
   information of one minibatch in AudioDataset.
2. After AudioDataLoader getting one minibatch from AudioDataset,
   AudioDataLoader calls its collate_fn(batch) to process this minibatch.

Input:
    Mixtured WJS0 tr, cv and tt path
Output:
    One batch at a time.
    Each inputs's shape is B x T
    Each targets's shape is B x C x T
"""

import json
import math
import os
import sys

import numpy as np
import torch
import torch.utils.data as data

import librosa

# Class for DNS Challenge dataset, assumes that audio clips are 30 seconds each
# class AudioDataset(data.Dataset):
#
#     def __init__(self, json_dir, batch_size, sample_rate=16000, segment=3.0, max_hours=None):
#         """
#         Args:
#             json_dir: directory including mix.json, s1.json and s2.json
#             segment: duration of audio segment, when set to -1, use full audio
#             max_hours: number of hours to load into AudioDataset. if set to 1 - use everything
#         xxx_infos is a list and each item is a tuple (wav_file, #samples)
#
#         """
#         super(AudioDataset, self).__init__()
#         mix_json = os.path.join(json_dir, 'mix.json')
#         clean_json = os.path.join(json_dir, 'clean.json')
#         noise_json = os.path.join(json_dir, 'noise.json')
#         with open(mix_json, 'r') as f:
#             mix_infos = json.load(f)
#         with open(clean_json, 'r') as f:
#             clean_infos = json.load(f)
#         with open(noise_json, 'r') as f:
#             noise_infos = json.load(f)
#         # sort it by #samples (impl bucket)
#         def sort(infos): return sorted(
#             infos, key=lambda info: int(info[0].split("_")[-1].split(".")[0]))  # splits to get fileid
#         sorted_mix_infos = sort(mix_infos)
#         sorted_clean_infos = sort(clean_infos)
#         sorted_noise_infos = sort(noise_infos)
#
#
#         # segment length and count dropped utts
#         segment_len = int(segment * sample_rate)  # 4s * 8000/s = 32000 samples
#         drop_utt, drop_len, total_utt, total_len = 0, 0, 0, 0
#         # Remove utts that are smaller than 4s or bigger than batch_size
#         for _, sample in sorted_mix_infos:  # Only counts, doesn't remove them
#             num_segments = math.ceil(sample / segment_len)
#             if sample < segment_len:
#                 drop_utt += 1
#                 drop_len += sample
#             else:
#                 total_len += sample  # Use the whole utterance
#                 total_utt += 1
#         print("Dropped {} utts({:.2f} h) which are shorter than {} samples".format(
#             drop_utt, drop_len/sample_rate/3600, segment_len))
#         print("{} utts, total number of undropped hours: {:.2f} hours".format(
#             total_utt, total_len/sample_rate/(3600)))
#
#         # generate minibach infomations
#         minibatch = []
#         i_audio = 0
#         curr_num_hours = 0
#         while i_audio < len(sorted_mix_infos):
#             num_segments = 0
#
#             # Assume that DNS dataset has 30sec per utterance.
#             utt_len = int(sorted_mix_infos[i_audio][1])
#             num_segments += math.ceil(utt_len / segment_len)
#             num_batches_in_utt = math.ceil(num_segments / batch_size)
#             for i_batch in range(num_batches_in_utt):
#                 # Create a number of batches from this utternace.
#                 part_mix, part_clean, part_noise = [], [], []
#                 part_mix.append(sorted_mix_infos[i_audio])
#                 part_clean.append(sorted_clean_infos[i_audio])
#                 part_noise.append(sorted_noise_infos[i_audio])
#                 # i_batch means the i_th batch from the utt (see collate_fn)
#                 minibatch.append([part_mix, part_clean, part_noise, i_batch,
#                                   sample_rate, segment_len, batch_size])
#             curr_num_hours += utt_len / sample_rate / 3600
#             i_audio += 1
#             if max_hours is not None and curr_num_hours > max_hours:
#                 break
#         self.minibatch = minibatch
#
#     def __getitem__(self, index):
#         return self.minibatch[index]
#
#     def __len__(self):
#         return len(self.minibatch)


class AudioDataset(data.Dataset):

    def __init__(self, json_dir, batch_size, sample_rate=16000, segment=3.0, max_hours=None):
        """
        Args:
            json_dir: directory including mix.json, s1.json and s2.json
            segment: duration of audio segment, when set to -1, use full audio
            max_hours: number of hours to load into AudioDataset. if set to 1 - use everything
        xxx_infos is a list and each item is a tuple (wav_file, #samples)

        """
        super(AudioDataset, self).__init__()
        mix_json = os.path.join(json_dir, 'mix.json')
        clean_json = os.path.join(json_dir, 'clean.json')
        noise_json = os.path.join(json_dir, 'noise.json')
        with open(mix_json, 'r') as f:
            mix_infos = json.load(f)
        with open(clean_json, 'r') as f:
            clean_infos = json.load(f)
        with open(noise_json, 'r') as f:
            noise_infos = json.load(f)
        # sort it by #samples (impl bucket)
        def sort(infos): return sorted(
            infos, key=lambda info: int(info[0].split("_")[-1].split(".")[0]))  # splits to get fileid
        sorted_mix_infos = sort(mix_infos)
        sorted_clean_infos = sort(clean_infos)
        sorted_noise_infos = sort(noise_infos)


        # segment length and count dropped utts
        segment_len = int(segment * sample_rate)  # 4s * 8000/s = 32000 samples
        drop_utt, drop_len, total_utt, total_len = 0, 0, 0, 0
        # Remove utts that are smaller than 4s or bigger than batch_size
        for _, sample in sorted_mix_infos:  # Only counts, doesn't remove them
            num_segments = math.ceil(sample / segment_len)
            if sample < segment_len:
                drop_utt += 1
                drop_len += sample
            else:
                total_len += sample  # Use the whole utterance
                total_utt += 1
        print("Dropped {} utts({:.2f} h) which are shorter than {} samples".format(
            drop_utt, drop_len/sample_rate/3600, segment_len))
        print("{} utts, total number of undropped hours: {:.2f} hours".format(
            total_utt, total_len/sample_rate/(3600)))

        # generate minibach infomations
        minibatch = []
        start = 0
        curr_num_hours = 0
        i_batch = 0  # Bandaid fix
        while True:
            num_segments = 0
            i_audio = start
            part_mix, part_clean, part_noise = [], [], []
            # Run until I created a full batch
            while num_segments < batch_size and i_audio < len(sorted_mix_infos):
                utt_len = int(sorted_mix_infos[i_audio][1])
                if utt_len >= segment_len:  # skip too short utt
                    num_segments += math.ceil(utt_len / segment_len)
                    # Ensure num_segments is less than batch_size

                    if num_segments > batch_size and start != i_audio:
                        # if num_segments of 1st audio > batch_size, skip it (it's bigger than batch_size alone)
                        break
                    part_mix.append(sorted_mix_infos[i_audio])
                    part_clean.append(sorted_clean_infos[i_audio])
                    part_noise.append(sorted_noise_infos[i_audio])

                    curr_num_hours += min(utt_len, segment_len * batch_size) / sample_rate / 3600
                i_audio += 1
            if len(part_mix) > 0:
                minibatch.append([part_mix, part_clean, part_noise, i_batch,
                                  sample_rate, segment_len, batch_size])

            if i_audio == len(sorted_mix_infos):
                break
            if max_hours is not None and curr_num_hours > max_hours:
                break
            start = i_audio
        self.minibatch = minibatch

    def __getitem__(self, index):
        return self.minibatch[index]

    def __len__(self):
        return len(self.minibatch)

class AudioDataLoader(data.DataLoader):
    """
    NOTE: just use batchsize=1 here, so drop_last=True makes no sense here.
    """

    def __init__(self, *args, **kwargs):
        super(AudioDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn


def _collate_fn(batch):
    """
    Args:
        batch: list, len(batch) = 1. See AudioDataset.__getitem__()
    Returns:
        mix_segments_pad: B x T, torch.Tensor
        segment_lengths : B, torch.Tentor
        sources_segments_pad: B x C x T, torch.Tensor
    """
    # batch should be located in list
    assert len(batch) == 1
    mix_segments, sources_segments = load_mixtures_and_sources(batch[0])
    # get batch of lengths of input sequences
    segment_lengths = np.array([mix.shape[0] for mix in mix_segments])

    # perform padding and convert to tensor
    pad_value = 0  # converts a list to tensor since pad_value =0
    mix_segments_pad = pad_list([torch.from_numpy(mix).float()
                             for mix in mix_segments], pad_value)
    segment_lengths = torch.from_numpy(segment_lengths)
    sources_segments_pad = pad_list([torch.from_numpy(s).float()
                            for s in sources_segments], pad_value)
    sources_segments_pad = sources_segments_pad.permute((0, 2, 1)).contiguous()   # N x T x C -> N x C x T
    return mix_segments_pad, segment_lengths, sources_segments_pad


# Eval data part
from preprocess import preprocess_one_dir


class EvalDataset(data.Dataset):
    def __init__(self, mix_dir, mix_json, batch_size, sample_rate=8000):
        """
        Args:
            mix_dir: directory including mixture wav files
            mix_json: json file including mixture wav files
        """
        super(EvalDataset, self).__init__()
        assert mix_dir != None or mix_json != None
        if mix_dir is not None:
            # Generate mix.json given mix_dir
            preprocess_one_dir(mix_dir, mix_dir, 'mix',
                               sample_rate=sample_rate)
            mix_json = os.path.join(mix_dir, 'mix.json')
        with open(mix_json, 'r') as f:
            mix_infos = json.load(f)
        # sort it by #samples (impl bucket)
        def sort(infos): return sorted(
            infos, key=lambda info: int(info[1]), reverse=True)
        sorted_mix_infos = sort(mix_infos)
        # generate minibach infomations
        minibatch = []
        start = 0
        while True:
            end = min(len(sorted_mix_infos), start + batch_size)
            minibatch.append([sorted_mix_infos[start:end],
                              sample_rate])
            if end == len(sorted_mix_infos):
                break
            start = end
        self.minibatch = minibatch

    def __getitem__(self, index):
        return self.minibatch[index]

    def __len__(self):
        return len(self.minibatch)


class EvalDataLoader(data.DataLoader):
    """
    NOTE: just use batchsize=1 here, so drop_last=True makes no sense here.
    """

    def __init__(self, *args, **kwargs):
        super(EvalDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn_eval


def _collate_fn_eval(batch):
    """
    Args:
        batch: list, len(batch) = 1. See AudioDataset.__getitem__()
    Returns:
        mixtures_pad: B x T, torch.Tensor
        ilens : B, torch.Tentor
        filenames: a list contain B strings
    """
    # batch should be located in list
    assert len(batch) == 1
    mixtures, filenames = load_mixtures(batch[0])

    # get batch of lengths of input sequences
    ilens = np.array([mix.shape[0] for mix in mixtures])

    # perform padding and convert to tensor
    pad_value = 0
    mixtures_pad = pad_list([torch.from_numpy(mix).float()
                             for mix in mixtures], pad_value)
    ilens = torch.from_numpy(ilens)
    return mixtures_pad, ilens, filenames


# ------------------------------ utils ------------------------------------
# Function for DNS Dataset, assumes clips of 30s
# def load_mixtures_and_sources(batch):
#     """
#     Each info include wav path and wav duration.
#     Returns:
#         mixtures: a list containing B items, each item is T np.ndarray
#         sources: a list containing B items, each item is T x C np.ndarray
#         T varies from item to item.
#     """
#     mix_segments, sources_segments = [], []  # After we cut audio wave into batch_size parts ( maybe 4 )
#     mix_infos, clean_infos, noise_infos, i_batch, sample_rate, segment_len, batch_size = batch
#     # for each utterance
#     for mix_info, clean_info, noise_info in zip(mix_infos, clean_infos, noise_infos):
#         mix_path = mix_info[0]
#         clean_path = clean_info[0]
#         noise_path = noise_info[0]
#         assert mix_info[1] == clean_info[1] and clean_info[1] == noise_info[1]
#         # read wav file
#         mix_wave, _ = librosa.load(mix_path, sr=sample_rate)
#         clean_wave, _ = librosa.load(clean_path, sr=sample_rate)
#         noise_wave, _ = librosa.load(noise_path, sr=sample_rate)
#         # merge s1 and s2
#         s12_waves = np.dstack((clean_wave, noise_wave))[0]  # T x C, C = 2
#         utt_len = mix_wave.shape[-1]
#
#         # create one segment
#         # Allow audio that is longer than batch_size to be used
#         start_index = i_batch * (batch_size * segment_len)
#         num_segments = math.ceil(utt_len / segment_len)
#         num_batches_in_utt = math.ceil(num_segments / batch_size)
#         last_batch_index = min(start_index + (batch_size-1) * segment_len +1, utt_len - segment_len + 1)
#         for i in range(start_index, last_batch_index, segment_len):
#             mix_segments.append(mix_wave[i:i+segment_len])
#             sources_segments.append(s12_waves[i:i+segment_len, :])
#         if utt_len % segment_len != 0 and i_batch +1 == num_batches_in_utt:
#             mix_segments.append(mix_wave[-segment_len:])  # last segment that isn't full
#             sources_segments.append(s12_waves[-segment_len:, :])
#
#     return mix_segments, sources_segments


def load_mixtures_and_sources(batch):
    """
    Each info include wav path and wav duration.
    Returns:
        mixtures: a list containing B items, each item is T np.ndarray
        sources: a list containing B items, each item is T x C np.ndarray
        T varies from item to item.
    """
    mix_segments, sources_segments = [], []  # After we cut audio wave into batch_size parts ( maybe 4 )
    mix_infos, clean_infos, noise_infos, i_batch, sample_rate, segment_len, batch_size = batch
    # for each utterance
    for mix_info, clean_info, noise_info in zip(mix_infos, clean_infos, noise_infos):
        mix_path = mix_info[0]
        clean_path = clean_info[0]
        noise_path = noise_info[0]
        assert mix_info[1] == clean_info[1] and clean_info[1] == noise_info[1]
        # read wav file
        mix_wave, _ = librosa.load(mix_path, sr=sample_rate)
        clean_wave, _ = librosa.load(clean_path, sr=sample_rate)
        noise_wave, _ = librosa.load(noise_path, sr=sample_rate)
        # merge s1 and s2
        s12_waves = np.dstack((clean_wave, noise_wave))[0]  # T x C, C = 2
        utt_len = mix_wave.shape[-1]

        # create one segment
        # Allow audio that is longer than batch_size to be used
        start_index = i_batch * (batch_size * segment_len)
        num_segments = math.ceil(utt_len / segment_len)
        num_batches_in_utt = math.ceil(num_segments / batch_size)
        last_batch_index = min(start_index + (batch_size-1) * segment_len +1, utt_len - segment_len + 1)
        for i in range(start_index, last_batch_index, segment_len):
            mix_segments.append(mix_wave[i:i+segment_len])
            sources_segments.append(s12_waves[i:i+segment_len, :])
        if utt_len % segment_len != 0 and i_batch +1 == num_batches_in_utt:
            mix_segments.append(mix_wave[-segment_len:])  # last segment that isn't full
            sources_segments.append(s12_waves[-segment_len:, :])

    return mix_segments, sources_segments

def load_mixtures(batch):
    """
    Returns:
        mixtures: a list containing B items, each item is T np.ndarray
        filenames: a list containing B strings
        T varies from item to item.
    """
    mixtures, filenames = [], []
    mix_infos, sample_rate = batch
    # for each utterance
    for mix_info in mix_infos:
        mix_path = mix_info[0]
        # read wav file
        mix, _ = librosa.load(mix_path, sr=sample_rate)
        mixtures.append(mix)
        filenames.append(mix_path)
    return mixtures, filenames


def pad_list(xs, pad_value):
    """ input is list with length 4
        output is a tensor of 4x32000 for mix and 4x3200x2 for sources, since pad = 0"""
    n_batch = len(xs)
    max_len = max(x.size(0) for x in xs)
    # xs[0].size()[1:] means put it only if dimension exists
    pad = xs[0].new(n_batch, max_len, * xs[0].size()[1:]).fill_(pad_value)
    for i in range(n_batch):
        pad[i, :xs[i].size(0)] = xs[i]
    return pad


if __name__ == "__main__":
    json_dir, batch_size = sys.argv[1:3]
    dataset = AudioDataset(json_dir, int(batch_size))
    data_loader = AudioDataLoader(dataset, batch_size=1,
                                  num_workers=4)
    for i, batch in enumerate(data_loader):
        mixtures, lens, sources = batch
        print(i)
        print(mixtures.size())
        print(sources.size())
        print(lens)
        if i < 10:
            print(mixtures)
            print(sources)
