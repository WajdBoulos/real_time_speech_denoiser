# Created on 2018/12
# Author: Kaituo XU

from itertools import permutations

import torch
import torch.nn.functional as F
import scipy.signal
import torchaudio
from .utils import arrange_batch, parse_audio

EPS = 1e-8


def cal_loss(source, estimate_source, source_lengths, device, features_model=None):
    """
    Args:
        source: [B, T], B is batch size
        estimate_source: [B, T]
        source_lengths: [B]
    """
    # assert features_model is not None
    # loss = calc_deep_feature_loss(source, estimate_source, source_lengths, features_model, device)
    loss = 0.0 - torch.mean(calc_si_sdr(source, estimate_source, source_lengths))
    # import GPUtil
    # GPUtil.showinitialization()
    return loss


def calc_si_sdr(source, estimate_source, source_lengths):
    """ SI-SDR for Speech Enhancement from paper https://arxiv.org/abs/1909.01019 """

    assert source.size() == estimate_source.size()
    B, T = source.size()

    # Step 1. Zero-mean norm
    num_samples = source_lengths.view(-1, 1).float()  # [B, 1]
    mean_target = torch.sum(source, dim=1, keepdim=True) / num_samples
    mean_estimate = torch.sum(estimate_source, dim=1, keepdim=True) / num_samples
    zero_mean_target = source - mean_target
    zero_mean_estimate = estimate_source - mean_estimate
    #
    cross_energy = torch.sum(zero_mean_target * zero_mean_estimate, dim=1, keepdim=True)
    target_energy = torch.sum(zero_mean_target ** 2, dim=1, keepdim=True) + EPS
    estimate_energy = torch.sum(zero_mean_estimate ** 2, dim=1, keepdim=True) + EPS
    # si_sdr = 10 * torch.log10(cross_energy/ (target_energy * estimate_energy - cross_energy) + EPS)
    alpha = cross_energy / target_energy
    si_sdr = torch.sum((alpha * zero_mean_target) ** 2, dim=1, keepdim=True) / \
             torch.sum((alpha * zero_mean_target - zero_mean_estimate) ** 2, dim=1, keepdim=True)
    si_sdr = 10 * torch.log10(si_sdr)
    return si_sdr


# def calc_deep_feature_loss(source, estimate_source, source_lengths, deep_features_model, device):
#     """
#     Calculates deep feature loss using the Wav2Vec Model.
#     Args:
#         source: [B, T], B is batch size
#         estimate_source: [B, T]
#         source_lengths: [B], each item is between [0, T]
#     """
#     # TODO: Make sure that output sigal behaves like signal from librosa.load
#     #   ie check Decoder output for clean signal
#     B, T = source.size()
#     deep_features_model = deep_features_model.to(device)
#
#     features_source = deep_features_model.feature_extractor(source)
#     features_source = deep_features_model.feature_aggregator(features_source)
#
#     features_estimate = deep_features_model.feature_extractor(estimate_source)
#     features_estimate = deep_features_model.feature_aggregator(features_estimate)
#
#     mse_loss = torch.nn.MSELoss()


if __name__ == "__main__":
    torch.manual_seed(123)
    B, C, T = 2, 3, 32000  # In this case B is the number of batches
    # fake data
    source = torch.randint(4, (B, C, T))
    estimate_source = torch.randint(4, (B, C, T))
    source[1, :, -3:] = 0
    estimate_source[1, :, -3:] = 0
    source_lengths = torch.LongTensor([T, T - 3])
    print('source', source)
    print('estimate_source', estimate_source)
    print('source_lengths', source_lengths)

