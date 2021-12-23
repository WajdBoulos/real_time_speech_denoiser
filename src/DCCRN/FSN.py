
#wfeature
import os

import librosa
import math
import numpy as np
import torch
import torch.nn as nn


def stft(y, n_fft, hop_length, win_length):
    """
    Args:
        y: [B, F, T]
        n_fft:
        hop_length:
        win_length:
        device:

    Returns:
        [B, F, T], **complex-valued** STFT coefficients

    """
    assert y.dim() == 2
    return torch.stft(
        y,
        n_fft,
        hop_length,
        win_length,
        window=torch.hann_window(n_fft).to(y.device),
        return_complex=True
    )


def istft(features, n_fft, hop_length, win_length, length=None, use_mag_phase=False):
    """
    Wrapper for the official torch.istft

    Args:
        features: [B, F, T, 2] (complex) or ([B, F, T], [B, F, T]) (mag and phase)
        n_fft:
        hop_length:
        win_length:
        device:
        length:
        use_mag_phase: use mag and phase as inputs of iSTFT

    Returns:
        [B, T]
    """
    if use_mag_phase:
        # (mag, phase) or [mag, phase]
        assert isinstance(features, tuple) or isinstance(features, list)
        mag, phase = features
        features = torch.stack([mag * torch.cos(phase), mag * torch.sin(phase)], dim=-1)

    return torch.istft(
        features,
        n_fft,
        hop_length,
        win_length,
        window=torch.hann_window(n_fft).to(features.device),
        length=length
    )


def mc_stft(y_s, n_fft, hop_length, win_length):
    """
    Multi-Channel STFT

    Shape:
        y_s: [B, C, T]

    Returns:
        complex_value: [B, C, F, T]
    """
    assert y_s.dim() == 3
    batch_size, num_channels, num_wav_samples = y_s.size()

    # [B * C, F, T] in C
    stft_coefficients = torch.stft(
        y_s.reshape(batch_size * num_channels, num_wav_samples),  # [B * C, T]
        n_fft=n_fft,
        hop_length=hop_length,
        window=torch.hann_window(win_length, device=y_s.device),
        win_length=win_length,
        return_complex=True
    )

    return stft_coefficients.reshape(batch_size, num_channels, stft_coefficients.shape[-2], stft_coefficients.shape[-1])


def mag_phase(complex_tensor):
    return torch.abs(complex_tensor), torch.angle(complex_tensor)


def norm_amplitude(y, scalar=None, eps=1e-6):
    if not scalar:
        scalar = np.max(np.abs(y)) + eps

    return y / scalar, scalar


def tailor_dB_FS(y, target_dB_FS=-25, eps=1e-6):
    rms = np.sqrt(np.mean(y ** 2))
    scalar = 10 ** (target_dB_FS / 20) / (rms + eps)
    y *= scalar
    return y, rms, scalar


def is_clipped(y, clipping_threshold=0.999):
    return any(np.abs(y) > clipping_threshold)


def load_wav(file, sr=16000):
    if len(file) == 2:
        return file[-1]
    else:
        return librosa.load(os.path.abspath(os.path.expanduser(file)), mono=False, sr=sr)[0]


def aligned_subsample(data_a, data_b, sub_sample_length):
    """
    Start from a random position and take a fixed-length segment from two speech samples

    Notes
        Only support one-dimensional speech signal (T,) and two-dimensional spectrogram signal (F, T)

        Only support subsample in the last axis.
    """
    assert data_a.shape[-1] == data_b.shape[-1], "Inconsistent dataset size."

    if data_a.shape[-1] > sub_sample_length:
        length = data_a.shape[-1]
        start = np.random.randint(length - sub_sample_length + 1)
        end = start + sub_sample_length
        # data_a = data_a[..., start: end]
        return data_a[..., start:end], data_b[..., start:end]
    elif data_a.shape[-1] < sub_sample_length:
        length = data_a.shape[-1]
        pad_size = sub_sample_length - length
        pad_width = [(0, 0)] * (data_a.ndim - 1) + [(0, pad_size)]
        data_a = np.pad(data_a, pad_width=pad_width, mode="constant", constant_values=0)
        data_b = np.pad(data_b, pad_width=pad_width, mode="constant", constant_values=0)
        return data_a, data_b
    else:
        return data_a, data_b


def subsample(data, sub_sample_length, start_position: int = -1, return_start_position=False):
    """
    Randomly select fixed-length data from

    Args:
        data: **one-dimensional data**
        sub_sample_length: how long
        start_position: If start index smaller than 0, randomly generate one index

    """
    assert np.ndim(data) == 1, f"Only support 1D data. The dim is {np.ndim(data)}"
    length = len(data)

    if length > sub_sample_length:
        if start_position < 0:
            start_position = np.random.randint(length - sub_sample_length)
        end = start_position + sub_sample_length
        data = data[start_position:end]
    elif length < sub_sample_length:
        data = np.append(data, np.zeros(sub_sample_length - length, dtype=np.float32))
    else:
        pass

    assert len(data) == sub_sample_length

    if return_start_position:
        return data, start_position
    else:
        return data


def overlap_cat(chunk_list, dim=-1):
    """
    按照 50% 的 overlap 沿着最后一个维度对 chunk_list 进行拼接

    Args:
        dim: 需要拼接的维度
        chunk_list(list): [[B, T], [B, T], ...]

    Returns:
        overlap 拼接后
    """
    overlap_output = []
    for i, chunk in enumerate(chunk_list):
        first_half, last_half = torch.split(chunk, chunk.size(-1) // 2, dim=dim)
        if i == 0:
            overlap_output += [first_half, last_half]
        else:
            overlap_output[-1] = (overlap_output[-1] + first_half) / 2
            overlap_output.append(last_half)

    overlap_output = torch.cat(overlap_output, dim=dim)
    return overlap_output


def activity_detector(audio, fs=16000, activity_threshold=0.13, target_level=-25, eps=1e-6):
    """
    Return the percentage of the time the audio signal is above an energy threshold

    Args:
        audio:
        fs:
        activity_threshold:
        target_level:
        eps:

    Returns:

    """
    audio, _, _ = tailor_dB_FS(audio, target_level)
    window_size = 50  # ms
    window_samples = int(fs * window_size / 1000)
    sample_start = 0
    cnt = 0
    prev_energy_prob = 0
    active_frames = 0

    a = -1
    b = 0.2
    alpha_rel = 0.05
    alpha_att = 0.8

    while sample_start < len(audio):
        sample_end = min(sample_start + window_samples, len(audio))
        audio_win = audio[sample_start:sample_end]
        frame_rms = 20 * np.log10(sum(audio_win ** 2) + eps)
        frame_energy_prob = 1. / (1 + np.exp(-(a + b * frame_rms)))

        if frame_energy_prob > prev_energy_prob:
            smoothed_energy_prob = frame_energy_prob * alpha_att + prev_energy_prob * (1 - alpha_att)
        else:
            smoothed_energy_prob = frame_energy_prob * alpha_rel + prev_energy_prob * (1 - alpha_rel)

        if smoothed_energy_prob > activity_threshold:
            active_frames += 1
        prev_energy_prob = frame_energy_prob
        sample_start += window_samples
        cnt += 1

    perc_active = active_frames / cnt
    return perc_active


def drop_band(input, num_groups=2):
    """
    Reduce computational complexity of the sub-band part in the FullSubNet model.

    Shapes:
        input: [B, C, F, T]
        return: [B, C, F // num_groups, T]
    """
    batch_size, _, num_freqs, _ = input.shape
    assert batch_size > num_groups, f"Batch size = {batch_size}, num_groups = {num_groups}. The batch size should larger than the num_groups."

    if num_groups <= 1:
        # No demand for grouping
        return input

    # Each sample must has the same number of the frequencies for parallel training.
    # Therefore, we need to drop those remaining frequencies in the high frequency part.
    if num_freqs % num_groups != 0:
        input = input[..., :(num_freqs - (num_freqs % num_groups)), :]
        num_freqs = input.shape[2]

    output = []
    for group_idx in range(num_groups):
        samples_indices = torch.arange(group_idx, batch_size, num_groups, device=input.device)
        freqs_indices = torch.arange(group_idx, num_freqs, num_groups, device=input.device)

        selected_samples = torch.index_select(input, dim=0, index=samples_indices)
        selected = torch.index_select(selected_samples, dim=2, index=freqs_indices)  # [B, C, F // num_groups, T]

        output.append(selected)

    return torch.cat(output, dim=0)


def init_stft_kernel(frame_len,
                     frame_hop,
                     num_fft=None,
                     window="sqrt_hann"):
    if window != "sqrt_hann":
        raise RuntimeError("Now only support sqrt hanning window in order "
                           "to make signal perfectly reconstructed")
    if not num_fft:
        # FFT points
        fft_size = 2 ** math.ceil(math.log2(frame_len))
    else:
        fft_size = num_fft
    # window [window_length]
    window = torch.hann_window(frame_len) ** 0.5
    S_ = 0.5 * (fft_size * fft_size / frame_hop) ** 0.5
    # window_length, F, 2 (real+imag)
    kernel = torch.rfft(torch.eye(fft_size) / S_, 1)[:frame_len]
    # 2, F, window_length
    kernel = torch.transpose(kernel, 0, 2) * window
    # 2F, 1, window_length
    kernel = torch.reshape(kernel, (fft_size + 2, 1, frame_len))
    return kernel


class CustomSTFTBase(nn.Module):
    """
    Base layer for (i)STFT
    NOTE:
        1) Recommend sqrt_hann window with 2**N frame length, because it
           could achieve perfect reconstruction after overlap-add
        2) Now haven't consider padding problems yet
    """

    def __init__(self,
                 frame_len,
                 frame_hop,
                 window="sqrt_hann",
                 num_fft=None):
        super(CustomSTFTBase, self).__init__()
        K = init_stft_kernel(
            frame_len,
            frame_hop,
            num_fft=num_fft,
            window=window)
        self.K = nn.Parameter(K, requires_grad=False)
        self.stride = frame_hop
        self.window = window

    def freeze(self):
        self.K.requires_grad = False

    def unfreeze(self):
        self.K.requires_grad = True

    def check_nan(self):
        num_nan = torch.sum(torch.isnan(self.K))
        if num_nan:
            raise RuntimeError(
                "detect nan in STFT kernels: {:d}".format(num_nan))

    def extra_repr(self):
        return "window={0}, stride={1}, requires_grad={2}, kernel_size={3[0]}x{3[2]}".format(
            self.window, self.stride, self.K.requires_grad, self.K.shape)


class CustomSTFT(CustomSTFTBase):
    """
    Short-time Fourier Transform as a Layer
    """

    def __init__(self, *args, **kwargs):
        super(CustomSTFT, self).__init__(*args, **kwargs)

    def forward(self, x):
        """
        Accept raw waveform and output magnitude and phase
        x: input signal, N x 1 x S or N x S
        m: magnitude, N x F x T
        p: phase, N x F x T
        """
        if x.dim() not in [2, 3]:
            raise RuntimeError("Expect 2D/3D tensor, but got {:d}D".format(
                x.dim()))
        self.check_nan()
        # if N x S, reshape N x 1 x S
        if x.dim() == 2:
            x = torch.unsqueeze(x, 1)
        # N x 2F x T
        c = torch.nn.functional.conv1d(x, self.K, stride=self.stride, padding=0)
        # N x F x T
        r, i = torch.chunk(c, 2, dim=1)
        m = (r ** 2 + i ** 2) ** 0.5
        p = torch.atan2(i, r)
        return m, p, r, i


class CustomISTFT(CustomSTFTBase):
    """
    Inverse Short-time Fourier Transform as a Layer
    """

    def __init__(self, *args, **kwargs):
        super(CustomISTFT, self).__init__(*args, **kwargs)

    def forward(self, m, p, squeeze=False):
        """
        Accept phase & magnitude and output raw waveform
        m, p: N x F x T
        s: N x C x S
        """
        if p.dim() != m.dim() or p.dim() not in [2, 3]:
            raise RuntimeError("Expect 2D/3D tensor, but got {:d}D".format(
                p.dim()))
        self.check_nan()
        # if F x T, reshape 1 x F x T
        if p.dim() == 2:
            p = torch.unsqueeze(p, 0)
            m = torch.unsqueeze(m, 0)
        r = m * torch.cos(p)
        i = m * torch.sin(p)
        # N x 2F x T
        c = torch.cat([r, i], dim=1)
        # N x 2F x T
        s = torch.nn.functional.conv_transpose1d(c, self.K, stride=self.stride, padding=0)
        if squeeze:
            s = torch.squeeze(s)
        return s


class ChannelWiseLayerNorm(nn.LayerNorm):
    """
    Channel wise layer normalization
    """

    def __init__(self, *args, **kwargs):
        super(ChannelWiseLayerNorm, self).__init__(*args, **kwargs)

    def forward(self, x):
        """
        x: BS x N x K
        """
        if x.dim() != 3:
            raise RuntimeError("{} accept 3D tensor as input".format(
                self.__name__))
        # BS x N x K => BS x K x N
        x = torch.transpose(x, 1, 2)
        x = super(ChannelWiseLayerNorm, self).forward(x)
        x = torch.transpose(x, 1, 2)
        return x


class DirectionalFeatureComputer(nn.Module):
    def __init__(
            self,
            n_fft,
            win_length,
            hop_length,
            input_features,
            mic_pairs,
            lps_channel,
            use_cos_IPD=True,
            use_sin_IPD=False,
            eps=1e-8
    ):
        super().__init__()
        self.eps = eps
        self.input_features = input_features

        # STFT setting
        self.stft = CustomSTFT(frame_len=win_length, frame_hop=hop_length, num_fft=n_fft)
        self.num_freqs = n_fft // 2 + 1

        # IPD setting
        self.mic_pairs = np.array(mic_pairs)
        self.num_mic_pairs = self.mic_pairs.shape[0]
        self.ipd_left = [t[0] for t in mic_pairs]
        self.ipd_right = [t[1] for t in mic_pairs]
        self.use_cos_IPD = use_cos_IPD
        self.use_sin_IPD = use_sin_IPD

        self.lps_channel = lps_channel

        self.directional_feature_dim = 0
        if 'LPS' in self.input_features:
            self.directional_feature_dim += self.num_freqs
            self.lps_layer_norm = ChannelWiseLayerNorm(self.num_freqs)

        if 'IPD' in self.input_features:
            self.directional_feature_dim += self.num_freqs * self.num_mic_pairs
            if self.use_sin_IPD:
                self.directional_feature_dim += self.num_freqs * self.num_mic_pairs

    def compute_ipd(self, phase):
        """
        Args
            phase: phase of shape [B, M, F, K]
        Returns
            IPD  of shape [B, I, F, K]
        """
        cos_ipd = torch.cos(phase[:, self.ipd_left] - phase[:, self.ipd_right])
        sin_ipd = torch.sin(phase[:, self.ipd_left] - phase[:, self.ipd_right])
        return cos_ipd, sin_ipd

    def forward(self, y):
        """
        Args:
            y: input mixture waveform with shape [B, M, T]

        Notes:
            B - batch_size
            M - num_channels
            C - num_speakers
            F - num_freqs
            T - seq_len or num_samples
            K - num_frames
            I - IPD feature_size

        Returns:
            Spatial features and directional features of shape [B, ?, K]
        """
        batch_size, num_channels, num_samples = y.shape
        y = y.view(-1, num_samples)  # [B * M, T]
        magnitude, phase, real, imag = self.stft(y)
        _, num_freqs, num_frames = phase.shape  # [B * M, F, K]

        magnitude = magnitude.view(batch_size, num_channels, num_freqs, num_frames)
        phase = phase.view(batch_size, num_channels, num_freqs, num_frames)
        real = real.view(batch_size, num_channels, num_freqs, num_frames)
        imag = imag.view(batch_size, num_channels, num_freqs, num_frames)

        directional_feature = []
        if "LPS" in self.input_features:
            lps = torch.log(magnitude[:, self.lps_channel, ...] ** 2 + self.eps)  # [B, F, K], the 4-th channel, which is counted from right to left.
            lps = self.lps_layer_norm(lps)
            directional_feature.append(lps)

        if "IPD" in self.input_features:
            cos_ipd, sin_ipd = self.compute_ipd(phase)  # [B, I, F, K]
            cos_ipd = cos_ipd.view(batch_size, -1, num_frames)  # [B, I * F, K]
            sin_ipd = sin_ipd.view(batch_size, -1, num_frames)
            directional_feature.append(cos_ipd)
            if self.use_sin_IPD:
                directional_feature.append(sin_ipd)

        directional_feature = torch.cat(directional_feature, dim=1)

        return directional_feature, magnitude, phase, real, imag


class ChannelDirectionalFeatureComputer(nn.Module):
    def __init__(
            self,
            n_fft,
            win_length,
            hop_length,
            input_features,
            mic_pairs,
            lps_channel,
            use_cos_IPD=True,
            use_sin_IPD=False,
            eps=1e-8
    ):
        super().__init__()
        self.eps = eps
        self.input_features = input_features

        # STFT setting
        self.stft = CustomSTFT(frame_len=win_length, frame_hop=hop_length, num_fft=n_fft)
        self.num_freqs = n_fft // 2 + 1

        # IPD setting
        self.mic_pairs = np.array(mic_pairs)
        self.num_mic_pairs = self.mic_pairs.shape[0]
        self.ipd_left = [t[0] for t in mic_pairs]
        self.ipd_right = [t[1] for t in mic_pairs]
        self.use_cos_IPD = use_cos_IPD
        self.use_sin_IPD = use_sin_IPD

        self.lps_channel = lps_channel

        self.directional_feature_dim = 0
        if 'LPS' in self.input_features:
            self.directional_feature_dim += 1

        if 'IPD' in self.input_features:
            self.directional_feature_dim += self.num_mic_pairs
            if self.use_sin_IPD:
                self.directional_feature_dim += self.num_mic_pairs

    def compute_ipd(self, phase):
        """
        Args
            phase: phase of shape [B, M, F, K]
        Returns
            IPD  pf shape [B, I, F, K]
        """
        cos_ipd = torch.cos(phase[:, self.ipd_left] - phase[:, self.ipd_right])
        sin_ipd = torch.sin(phase[:, self.ipd_left] - phase[:, self.ipd_right])
        return cos_ipd, sin_ipd

    def forward(self, y):
        """
        Args:
            y: input mixture waveform with shape [B, M, T]

        Notes:
            B - batch_size
            M - num_channels
            C - num_speakers
            F - num_freqs
            T - seq_len or num_samples
            K - num_frames
            I - IPD feature_size

        Returns:
            Spatial features and directional features of shape [B, ?, K]
        """
        batch_size, num_channels, num_samples = y.shape
        y = y.view(-1, num_samples)  # [B * M, T]
        magnitude, phase, real, imag = self.stft(y)
        _, num_freqs, num_frames = phase.shape  # [B * M, F, K]

        magnitude = magnitude.view(batch_size, num_channels, num_freqs, num_frames)
        phase = phase.view(batch_size, num_channels, num_freqs, num_frames)
        real = real.view(batch_size, num_channels, num_freqs, num_frames)
        imag = imag.view(batch_size, num_channels, num_freqs, num_frames)

        directional_feature = []
        if "LPS" in self.input_features:
            lps = torch.log(magnitude[:, self.lps_channel, ...] ** 2 + self.eps)  # [B, F, K], the 4-th channel, which is counted from right to left.
            lps = lps[:, None, ...]
            directional_feature.append(lps)

        if "IPD" in self.input_features:
            cos_ipd, sin_ipd = self.compute_ipd(phase)  # [B, I, F, K]
            directional_feature.append(cos_ipd)

            if self.use_sin_IPD:
                directional_feature.append(sin_ipd)

        directional_feature = torch.cat(directional_feature, dim=1)

        # [B, C + I, F, T], [B, C, F, T], [B, C, F, T]
        return directional_feature, magnitude, phase, real, imag


if __name__ == '__main__':
    ipt = torch.rand(70, 1, 257, 200)
    print(drop_band(ipt, 8).shape)

#wconstant

import math
import numpy as np
import torch

NEG_INF = torch.finfo(torch.float32).min
PI = math.pi
SOUND_SPEED = 343  # m/s
EPSILON = np.finfo(np.float32).eps
MAX_INT16 = np.iinfo(np.int16).max

#wsequencemodel

import torch
import torch.nn as nn


class SequenceModel(nn.Module):
    def __init__(
            self,
            input_size,
            output_size,
            hidden_size,
            num_layers,
            bidirectional,
            sequence_model="GRU",
            output_activate_function="Tanh"
    ):
        """
        序列模型，可选 LSTM 或 CRN，支持子带输入

        Args:
            input_size: 每帧输入特征大小
            output_size: 每帧输出特征大小
            hidden_size: 序列模型隐层单元数量
            num_layers:  层数
            bidirectional: 是否为双向
            sequence_model: LSTM | GRU
            output_activate_function: Tanh | ReLU
        """
        super().__init__()
        # Sequence layer
        if sequence_model == "LSTM":
            self.sequence_model = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
            )
        elif sequence_model == "GRU":
            self.sequence_model = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
            )
        else:
            raise NotImplementedError(f"Not implemented {sequence_model}")

        # Fully connected layer
        if bidirectional:
            self.fc_output_layer = nn.Linear(hidden_size * 2, output_size)
        else:
            self.fc_output_layer = nn.Linear(hidden_size, output_size)

        # Activation function layer
        if output_activate_function:
            if output_activate_function == "Tanh":
                self.activate_function = nn.Tanh()
            elif output_activate_function == "ReLU":
                self.activate_function = nn.ReLU()
            elif output_activate_function == "ReLU6":
                self.activate_function = nn.ReLU6()
            else:
                raise NotImplementedError(f"Not implemented activation function {self.activate_function}")

        self.output_activate_function = output_activate_function

    def forward(self, x):
        """
        Args:
            x: [B, F, T]
        Returns:
            [B, F, T]
        """
        assert x.dim() == 3
        self.sequence_model.flatten_parameters()

        # contiguous 使元素在内存中连续，有利于模型优化，但分配了新的空间
        # 建议在网络开始大量计算前使用一下
        x = x.permute(0, 2, 1).contiguous()  # [B, F, T] => [B, T, F]
        o, _ = self.sequence_model(x)
        o = self.fc_output_layer(o)
        if self.output_activate_function:
            o = self.activate_function(o)
        o = o.permute(0, 2, 1).contiguous()  # [B, T, F] => [B, F, T]
        return o


def _print_networks(nets: list):
    print(f"This project contains {len(nets)} networks, the number of the parameters: ")
    params_of_all_networks = 0
    for i, net in enumerate(nets, start=1):
        params_of_network = 0
        for param in net.parameters():
            params_of_network += param.numel()

        print(f"\tNetwork {i}: {params_of_network / 1e6} million.")
        params_of_all_networks += params_of_network

    print(f"The amount of parameters in the project is {params_of_all_networks / 1e6} million.")


if __name__ == '__main__':
    import datetime

    with torch.no_grad():
        ipt = torch.rand(1, 257, 1000)
        model = SequenceModel(
            input_size=257,
            output_size=2,
            hidden_size=512,
            bidirectional=False,
            num_layers=3,
            sequence_model="LSTM"
        )

        start = datetime.datetime.now()
        opt = model(ipt)
        end = datetime.datetime.now()
        print(f"{end - start}")
        _print_networks([model, ])


#wbasemodel

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn import functional

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

    @staticmethod
    def unfold(input, num_neighbor):
        """
        Along with the frequency dim, split overlapped sub band units from spectrogram.

        Args:
            input: [B, C, F, T]
            num_neighbor:

        Returns:
            [B, N, C, F_s, T], F 为子频带的频率轴大小, e.g. [2, 161, 1, 19, 200]
        """
        assert input.dim() == 4, f"The dim of input is {input.dim()}. It should be four dim."
        batch_size, num_channels, num_freqs, num_frames = input.size()

        if num_neighbor < 1:
            # No change for the input
            return input.permute(0, 2, 1, 3).reshape(batch_size, num_freqs, num_channels, 1, num_frames)

        output = input.reshape(batch_size * num_channels, 1, num_freqs, num_frames)
        sub_band_unit_size = num_neighbor * 2 + 1

        # Pad to the top and bottom
        output = functional.pad(output, [0, 0, num_neighbor, num_neighbor], mode="reflect")

        output = functional.unfold(output, (sub_band_unit_size, num_frames))
        assert output.shape[-1] == num_freqs, f"n_freqs != N (sub_band), {num_freqs} != {output.shape[-1]}"

        # Split the dim of the unfolded feature
        output = output.reshape(batch_size, num_channels, sub_band_unit_size, num_frames, num_freqs)
        output = output.permute(0, 4, 1, 2, 3).contiguous()

        return output

    @staticmethod
    def _reduce_complexity_separately(sub_band_input, full_band_output, device):
        """

        Args:
            sub_band_input: [60, 257, 1, 33, 200]
            full_band_output: [60, 257, 1, 3, 200]
            device:

        Notes:
            1. 255 and 256 freq not able to be trained
            2. batch size 应该被 3 整除，否则最后一部分 batch 内的频率无法很好的训练

        Returns:
            [60, 85, 1, 36, 200]
        """
        batch_size = full_band_output.shape[0]
        n_freqs = full_band_output.shape[1]
        sub_batch_size = batch_size // 3
        final_selected = []

        for idx in range(3):
            # [0, 60) => [0, 20)
            sub_batch_indices = torch.arange(idx * sub_batch_size, (idx + 1) * sub_batch_size, device=device)
            full_band_output_sub_batch = torch.index_select(full_band_output, dim=0, index=sub_batch_indices)
            sub_band_output_sub_batch = torch.index_select(sub_band_input, dim=0, index=sub_batch_indices)

            # Avoid to use padded value (first freq and last freq)
            # i = 0, (1, 256, 3) = [1, 4, ..., 253]
            # i = 1, (2, 256, 3) = [2, 5, ..., 254]
            # i = 2, (3, 256, 3) = [3, 6, ..., 255]
            freq_indices = torch.arange(idx + 1, n_freqs - 1, step=3, device=device)
            full_band_output_sub_batch = torch.index_select(full_band_output_sub_batch, dim=1, index=freq_indices)
            sub_band_output_sub_batch = torch.index_select(sub_band_output_sub_batch, dim=1, index=freq_indices)

            # ([30, 85, 1, 33 200], [30, 85, 1, 3, 200]) => [30, 85, 1, 36, 200]

            final_selected.append(torch.cat([sub_band_output_sub_batch, full_band_output_sub_batch], dim=-2))

        return torch.cat(final_selected, dim=0)

    @staticmethod
    def sband_forgetting_norm(input, train_sample_length):
        """
        与 forgetting norm相同，但使用拼接后模型的中间频带来计算均值
        效果不好
        Args:
            input:
            train_sample_length:

        Returns:

        """
        assert input.ndim == 3
        batch_size, n_freqs, n_frames = input.size()

        eps = 1e-10
        alpha = (train_sample_length - 1) / (train_sample_length + 1)
        mu = 0
        mu_list = []

        for idx in range(input.shape[-1]):
            if idx < train_sample_length:
                alp = torch.min(torch.tensor([(idx - 1) / (idx + 1), alpha]))
                mu = alp * mu + (1 - alp) * torch.mean(input[:, :, idx], dim=1).reshape(batch_size, 1)  # [B, 1]
            else:
                mu = alpha * mu + (1 - alpha) * input[:, (n_freqs // 2 - 1), idx].reshape(batch_size, 1)

            mu_list.append(mu)

            # print("input", input[:, :, idx].min(), input[:, :, idx].max(), input[:, :, idx].mean())
            # print(f"alp {idx}: ", alp)
            # print(f"mu {idx}: {mu[128, 0]}")

        mu = torch.stack(mu_list, dim=-1)  # [B, 1, T]
        input = input / (mu + eps)
        return input

    @staticmethod
    def forgetting_norm(input, sample_length_in_training):
        """
        输入为三维，通过不断估计邻近的均值来作为当前 norm 时的均值

        Args:
            input: [B, F, T]
            sample_length_in_training: 训练时的长度，用于计算平滑因子

        Returns:

        """
        assert input.ndim == 3
        batch_size, n_freqs, n_frames = input.size()
        eps = 1e-10
        mu = 0
        alpha = (sample_length_in_training - 1) / (sample_length_in_training + 1)

        mu_list = []
        for idx in range(input.shape[-1]):
            if idx < sample_length_in_training:
                alp = torch.min(torch.tensor([(idx - 1) / (idx + 1), alpha]))
                mu = alp * mu + (1 - alp) * torch.mean(input[:, :, idx], dim=1).reshape(batch_size, 1)  # [B, 1]
            else:
                current_frame_mu = torch.mean(input[:, :, idx], dim=1).reshape(batch_size, 1)  # [B, 1]
                mu = alpha * mu + (1 - alpha) * current_frame_mu

            mu_list.append(mu)

            # print("input", input[:, :, idx].min(), input[:, :, idx].max(), input[:, :, idx].mean())
            # print(f"alp {idx}: ", alp)
            # print(f"mu {idx}: {mu[128, 0]}")

        mu = torch.stack(mu_list, dim=-1)  # [B, 1, T]
        input = input / (mu + eps)
        return input

    @staticmethod
    def hybrid_norm(input, sample_length_in_training=192):
        """
        Args:
            input: [B, F, T]
            sample_length_in_training:

        Returns:
            [B, F, T]
        """
        assert input.ndim == 3
        device = input.device
        data_type = input.dtype
        batch_size, n_freqs, n_frames = input.size()
        eps = 1e-10

        mu = 0
        alpha = (sample_length_in_training - 1) / (sample_length_in_training + 1)
        mu_list = []
        for idx in range(input.shape[-1]):
            if idx < sample_length_in_training:
                alp = torch.min(torch.tensor([(idx - 1) / (idx + 1), alpha]))
                mu = alp * mu + (1 - alp) * torch.mean(input[:, :, idx], dim=1).reshape(batch_size, 1)  # [B, 1]
                mu_list.append(mu)
            else:
                break
        initial_mu = torch.stack(mu_list, dim=-1)  # [B, 1, T]

        step_sum = torch.sum(input, dim=1)  # [B, T]
        cumulative_sum = torch.cumsum(step_sum, dim=-1)  # [B, T]

        entry_count = torch.arange(n_freqs, n_freqs * n_frames + 1, n_freqs, dtype=data_type, device=device)
        entry_count = entry_count.reshape(1, n_frames)  # [1, T]
        entry_count = entry_count.expand_as(cumulative_sum)  # [1, T] => [B, T]

        cum_mean = cumulative_sum / entry_count  # B, T

        cum_mean = cum_mean.reshape(batch_size, 1, n_frames)  # [B, 1, T]

        # print(initial_mu[0, 0, :50])
        # print("-"*60)
        # print(cum_mean[0, 0, :50])
        cum_mean[:, :, :sample_length_in_training] = initial_mu

        return input / (cum_mean + eps)

    @staticmethod
    def offline_laplace_norm(input):
        """

        Args:
            input: [B, C, F, T]

        Returns:
            [B, C, F, T]
        """
        # utterance-level mu
        mu = torch.mean(input, dim=(1, 2, 3), keepdim=True)

        normed = input / (mu + 1e-5)

        return normed

    @staticmethod
    def cumulative_laplace_norm(input):
        """

        Args:
            input: [B, C, F, T]

        Returns:

        """
        batch_size, num_channels, num_freqs, num_frames = input.size()
        input = input.reshape(batch_size * num_channels, num_freqs, num_frames)

        step_sum = torch.sum(input, dim=1)  # [B * C, F, T] => [B, T]
        cumulative_sum = torch.cumsum(step_sum, dim=-1)  # [B, T]

        entry_count = torch.arange(
            num_freqs,
            num_freqs * num_frames + 1,
            num_freqs,
            dtype=input.dtype,
            device=input.device
        )
        entry_count = entry_count.reshape(1, num_frames)  # [1, T]
        entry_count = entry_count.expand_as(cumulative_sum)  # [1, T] => [B, T]

        cumulative_mean = cumulative_sum / entry_count  # B, T
        cumulative_mean = cumulative_mean.reshape(batch_size * num_channels, 1, num_frames)

        normed = input / (cumulative_mean + EPSILON)

        return normed.reshape(batch_size, num_channels, num_freqs, num_frames)

    @staticmethod
    def offline_gaussian_norm(input):
        """
        Zero-Norm
        Args:
            input: [B, C, F, T]

        Returns:
            [B, C, F, T]
        """
        mu = torch.mean(input, dim=(1, 2, 3), keepdim=True)
        std = torch.std(input, dim=(1, 2, 3), keepdim=True)

        normed = (input - mu) / (std + 1e-5)

        return normed

    @staticmethod
    def cumulative_layer_norm(input):
        """
        Online zero-norm

        Args:
            input: [B, C, F, T]

        Returns:
            [B, C, F, T]
        """
        batch_size, num_channels, num_freqs, num_frames = input.size()
        input = input.reshape(batch_size * num_channels, num_freqs, num_frames)

        step_sum = torch.sum(input, dim=1)  # [B * C, F, T] => [B, T]
        step_pow_sum = torch.sum(torch.square(input), dim=1)

        cumulative_sum = torch.cumsum(step_sum, dim=-1)  # [B, T]
        cumulative_pow_sum = torch.cumsum(step_pow_sum, dim=-1)  # [B, T]

        entry_count = torch.arange(
            num_freqs,
            num_freqs * num_frames + 1,
            num_freqs,
            dtype=input.dtype,
            device=input.device
        )
        entry_count = entry_count.reshape(1, num_frames)  # [1, T]
        entry_count = entry_count.expand_as(cumulative_sum)  # [1, T] => [B, T]

        cumulative_mean = cumulative_sum / entry_count  # [B, T]
        cumulative_var = (cumulative_pow_sum - 2 * cumulative_mean * cumulative_sum) / entry_count + cumulative_mean.pow(2)  # [B, T]
        cumulative_std = torch.sqrt(cumulative_var + EPSILON)  # [B, T]

        cumulative_mean = cumulative_mean.reshape(batch_size * num_channels, 1, num_frames)
        cumulative_std = cumulative_std.reshape(batch_size * num_channels, 1, num_frames)

        normed = (input - cumulative_mean) / cumulative_std

        return normed.reshape(batch_size, num_channels, num_freqs, num_frames)

    def norm_wrapper(self, norm_type: str):
        if norm_type == "offline_laplace_norm":
            norm = self.offline_laplace_norm
        elif norm_type == "cumulative_laplace_norm":
            norm = self.cumulative_laplace_norm
        elif norm_type == "offline_gaussian_norm":
            norm = self.offline_gaussian_norm
        elif norm_type == "cumulative_layer_norm":
            norm = self.cumulative_layer_norm
        else:
            raise NotImplementedError("You must set up a type of Norm. "
                                      "e.g. offline_laplace_norm, cumulative_laplace_norm, forgetting_norm, etc.")
        return norm

    def weight_init(self, m):
        """
        Usage:
            model = Model()
            model.apply(weight_init)
        """
        if isinstance(m, nn.Conv1d):
            init.normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.Conv3d):
            init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.ConvTranspose1d):
            init.normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.ConvTranspose2d):
            init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.ConvTranspose3d):
            init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.BatchNorm1d):
            init.normal_(m.weight.data, mean=1, std=0.02)
            init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.normal_(m.weight.data, mean=1, std=0.02)
            init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm3d):
            init.normal_(m.weight.data, mean=1, std=0.02)
            init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.Linear):
            init.xavier_normal_(m.weight.data)
            init.normal_(m.bias.data)
        elif isinstance(m, nn.LSTM):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    init.orthogonal_(param.data)
                else:
                    init.normal_(param.data)
        elif isinstance(m, nn.LSTMCell):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    init.orthogonal_(param.data)
                else:
                    init.normal_(param.data)
        elif isinstance(m, nn.GRU):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    init.orthogonal_(param.data)
                else:
                    init.normal_(param.data)
        elif isinstance(m, nn.GRUCell):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    init.orthogonal_(param.data)
                else:
                    init.normal_(param.data)




#wfullsubnet

import torch
from torch.nn import functional
import sys
#sys.path.append("..") # Adds higher directory to python modules path.

#from audio_zen.acoustics.feature import drop_band
#from audio_zen.model.base_model import BaseModel
#from audio_zen.model.module.sequence_model import SequenceModel
import cProfile

def runFB(fb, arg):
    a = fb(arg)
    return a
def runSB(fb, arg):
    return fb(arg)

class Model(BaseModel):
    def __init__(self,
                 num_freqs,
                 look_ahead,
                 sequence_model,
                 fb_num_neighbors,
                 sb_num_neighbors,
                 fb_output_activate_function,
                 sb_output_activate_function,
                 fb_model_hidden_size,
                 sb_model_hidden_size,
                 norm_type="offline_laplace_norm",
                 num_groups_in_drop_band=2,
                 weight_init=True,
                 ):
        """
        FullSubNet model (cIRM mask)

        Args:
            num_freqs: Frequency dim of the input
            sb_num_neighbors: Number of the neighbor frequencies in each side
            look_ahead: Number of use of the future frames
            sequence_model: Chose one sequence model as the basic model (GRU, LSTM)
        """
        super().__init__()
        assert sequence_model in ("GRU", "LSTM"), f"{self.__class__.__name__} only support GRU and LSTM."

        self.fb_model = SequenceModel(
            input_size=num_freqs,
            output_size=num_freqs,
            hidden_size=fb_model_hidden_size,
            num_layers=2,
            bidirectional=False,
            sequence_model=sequence_model,
            output_activate_function=fb_output_activate_function
        )

        self.sb_model = SequenceModel(
            input_size=(sb_num_neighbors * 2 + 1) + (fb_num_neighbors * 2 + 1),
            output_size=2,
            hidden_size=sb_model_hidden_size,
            num_layers=2,
            bidirectional=False,
            sequence_model=sequence_model,
            output_activate_function=sb_output_activate_function
        )

        self.sb_num_neighbors = sb_num_neighbors
        self.fb_num_neighbors = fb_num_neighbors
        self.look_ahead = look_ahead
        self.norm = self.norm_wrapper(norm_type)
        self.num_groups_in_drop_band = num_groups_in_drop_band

        if weight_init:
            self.apply(self.weight_init)

    def forward(self, noisy_mag):
        """
        Args:
            noisy_mag: noisy magnitude spectrogram

        Returns:
            The real part and imag part of the enhanced spectrogram

        Shapes:
            noisy_mag: [B, 1, F, T]
            return: [B, 2, F, T]
        """
        assert noisy_mag.dim() == 4
        noisy_mag = functional.pad(noisy_mag, [0, self.look_ahead])  # Pad the look ahead
        batch_size, num_channels, num_freqs, num_frames = noisy_mag.size()
        assert num_channels == 1, f"{self.__class__.__name__} takes the mag feature as inputs."

        # Fullband model
        fb_input = self.norm(noisy_mag).reshape(batch_size, num_channels * num_freqs, num_frames)
        #pr = cProfile.Profile()
        #pr.enable()

        fb_output = runFB(self.fb_model, fb_input).reshape(batch_size, 1, num_freqs, num_frames)


        #pr.disable()
        #pr.print_stats(sort='time')

        # Unfold the output of the fullband model, [B, N=F, C, F_f, T]
        fb_output_unfolded = self.unfold(fb_output, num_neighbor=self.fb_num_neighbors)
        fb_output_unfolded = fb_output_unfolded.reshape(batch_size, num_freqs, self.fb_num_neighbors * 2 + 1, num_frames)

        # Unfold noisy input, [B, N=F, C, F_s, T]
        noisy_mag_unfolded = self.unfold(noisy_mag, num_neighbor=self.sb_num_neighbors)
        noisy_mag_unfolded = noisy_mag_unfolded.reshape(batch_size, num_freqs, self.sb_num_neighbors * 2 + 1, num_frames)

        # Concatenation, [B, F, (F_s + F_f), T]
        sb_input = torch.cat([noisy_mag_unfolded, fb_output_unfolded], dim=2)
        sb_input = self.norm(sb_input)

        # Speeding up training without significant performance degradation. These will be updated to the paper later.
        if batch_size > 1:
            sb_input = drop_band(sb_input.permute(0, 2, 1, 3), num_groups=self.num_groups_in_drop_band)  # [B, (F_s + F_f), F//num_groups, T]
            num_freqs = sb_input.shape[2]
            sb_input = sb_input.permute(0, 2, 1, 3)  # [B, F//num_groups, (F_s + F_f), T]

        sb_input = sb_input.reshape(
            batch_size * num_freqs,
            (self.sb_num_neighbors * 2 + 1) + (self.fb_num_neighbors * 2 + 1),
            num_frames
        )

        # [B * F, (F_s + F_f), T] => [B * F, 2, T] => [B, F, 2, T]
        #pr = cProfile.Profile()
        #pr.enable()

        sb_mask = runSB(self.sb_model, sb_input)


        #pr.disable()
        #pr.print_stats(sort='time')
        sb_mask = sb_mask.reshape(batch_size, num_freqs, 2, num_frames).permute(0, 2, 1, 3).contiguous()

        output = sb_mask[:, :, :, self.look_ahead:]
        return output

    @classmethod
    def load_model(cls, path):
        # Load to CPU
        package = torch.load(path, map_location=lambda storage, loc: storage)
        model = cls.load_model_from_package(package)
        return model

    @classmethod
    def load_model_from_package(cls, package):
        model.load_state_dict(package['state_dict'])

        return model

    @staticmethod
    def serialize(model, optimizer, epoch, tr_loss=None, cv_loss=None):
        package = {
            # state
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
            'epoch': epoch
        }
        if tr_loss is not None:
            package['tr_loss'] = tr_loss
            package['cv_loss'] = cv_loss
        return package


if __name__ == "__main__":
    import datetime

    with torch.no_grad():
        model = Model(
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
        ipt = torch.rand(1, 800)  # 1.6s
        ipt_len = ipt.shape[-1]
        # 1000 frames (16s) - 5.65s (35.31%，纯模型) - 5.78s
        # 500 frames (8s) - 3.05s (38.12%，纯模型) - 3.04s
        # 200 frames (3.2s) - 1.19s (37.19%，纯模型) - 1.20s
        # 100 frames (1.6s) - 0.62s (38.75%，纯模型) - 0.65s
        start = datetime.datetime.now()

        complex_tensor = torch.stft(ipt, n_fft=512, hop_length=256)
        mag = (complex_tensor.pow(2.).sum(-1) + 1e-8).pow(0.5 * 1.0).unsqueeze(1)
        print(f"STFT: {datetime.datetime.now() - start}, {mag.shape}")

        enhanced_complex_tensor = model(mag).detach().permute(0, 2, 3, 1)
        print(enhanced_complex_tensor.shape)
        print(f"Model Inference: {datetime.datetime.now() - start}")

        enhanced = torch.istft(enhanced_complex_tensor, 512, 256, length=ipt_len)
        print(f"iSTFT: {datetime.datetime.now() - start}")

        print(f"{datetime.datetime.now() - start}")
        ipt = torch.rand(3, 1, 257, 200)
        print(model(ipt).shape)
