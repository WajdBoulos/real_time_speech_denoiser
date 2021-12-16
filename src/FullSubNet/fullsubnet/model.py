import torch
from torch.nn import functional
import sys
#sys.path.append("..") # Adds higher directory to python modules path.

#from audio_zen.acoustics.feature import drop_band
#from audio_zen.model.base_model import BaseModel
#from audio_zen.model.module.sequence_model import SequenceModel

#
from src.FullSubNet.audio_zen.acoustics.feature import drop_band
from src.FullSubNet.audio_zen.model.base_model import BaseModel
from src.FullSubNet.audio_zen.model.module.sequence_model import ComplexSequenceModel
import cProfile


def runFB(fb, arg):
    a = fb(arg)
    return a


def runSB(fb, arg):
    return fb(arg)


def decompress_cIRM(mask, K=10, limit=9.9):
    mask = limit * (mask >= limit) - limit * (mask <= -limit) + mask * (torch.abs(mask) < limit)
    mask = -K * torch.log((K - mask) / (K + mask))
    return mask


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
                 device="cuda",
                 decompress_mask=True
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

        self.fb_model = ComplexSequenceModel(
            input_size=num_freqs,
            output_size=num_freqs,
            hidden_size=fb_model_hidden_size,
            num_layers=2,
            bidirectional=False,
            sequence_model=sequence_model,
            output_activate_function=fb_output_activate_function
        )

        self.sb_model = ComplexSequenceModel(
            input_size=(sb_num_neighbors * 2 + 1) + (fb_num_neighbors * 2 + 1),
            output_size=1,
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
        self.device = device
        self.decompress_mask = decompress_mask

        self.to(device)

        if weight_init:
            self.apply(self.weight_init)

    def forward(self, noisy_sound: torch.tensor):
        """
        Args:
            noisy_sound: Batched sound tensor input

        Returns:
            clean batched sound tensor

        Shapes:
            noisy_sound: [B, T]
            return: [B, T]
        """
        assert noisy_sound.dim() == 2, f"{self.__class__.__name__} takes the input as real."

        noisy_sound = noisy_sound.to(self.device)
        noisy_complex = torch.stft(noisy_sound, n_fft=512, hop_length=256, window=torch.hann_window(512).to(self.device), return_complex=False)
        # [B, F, T, C] -> [B, C, F, T]
        noisy_complex = noisy_complex.permute(0,3,1,2).contiguous()

        noisy_padded_complex = functional.pad(noisy_complex, [0, self.look_ahead])  # Pad the look ahead
        batch_size, num_channels, num_freqs, num_frames = noisy_padded_complex.size()
        assert num_channels == 2

        # Fullband model
        fb_input = self.norm(noisy_padded_complex).reshape(batch_size, num_channels, num_freqs, num_frames)
        #pr = cProfile.Profile()
        #pr.enable()

        assert fb_input.dim() == 4
        fb_output = runFB(self.fb_model, fb_input).reshape(batch_size, num_channels, num_freqs, num_frames)

        #pr.disable()
        #pr.print_stats(sort='time')

        # Unfold the output of the fullband model, [B, N=F, C, F_f, T]
        fb_output_unfolded = self.unfold(fb_output, num_neighbor=self.fb_num_neighbors)
        fb_output_unfolded = fb_output_unfolded.reshape(batch_size, num_freqs, num_channels, self.fb_num_neighbors * 2 + 1, num_frames)

        # Unfold noisy input, [B, N=F, C, F_s, T]
        noisy_complex_unfolded = self.unfold(noisy_padded_complex, num_neighbor=self.sb_num_neighbors)
        noisy_complex_unfolded = noisy_complex_unfolded.reshape(batch_size, num_freqs, num_channels, self.sb_num_neighbors * 2 + 1, num_frames)

        # Concatenation, [B, F, C, (F_s + F_f), T]
        sb_input = torch.cat([noisy_complex_unfolded, fb_output_unfolded], dim=3)
        sb_input = self.norm(sb_input).reshape(batch_size, num_freqs, num_channels, (self.fb_num_neighbors * 2 + 1) +
                                               (self.sb_num_neighbors * 2 + 1), num_frames)

        # Speeding up training without significant performance degradation. These will be updated to the paper later.
        #TODO: enable the bellow optimization to support complex processing
        #if batch_size > 1:
        #    sb_input = drop_band(sb_input.permute(0, 2, 1, 3), num_groups=self.num_groups_in_drop_band)  # [B, (F_s + F_f), F//num_groups, T]
        #    num_freqs = sb_input.shape[2]
        #    sb_input = sb_input.permute(0, 2, 1, 3)  # [B, F//num_groups, (F_s + F_f), T]

        # [B * F, C, (F_s + F_f), T]
        sb_input = sb_input.reshape(
            batch_size * num_freqs, num_channels,
            (self.sb_num_neighbors * 2 + 1) + (self.fb_num_neighbors * 2 + 1),
            num_frames
        )

        # [B * F, C, (F_s + F_f), T] => [B * F, 2, T] => [B, F, 2, T]
        #pr = cProfile.Profile()
        #pr.enable()
        sb_mask = runSB(self.sb_model, sb_input)


        #pr.disable()
        #pr.print_stats(sort='time')
        # [B, F, 2, T] => [B, 2, F, T]
        sb_mask = sb_mask.reshape(batch_size, num_freqs, 2, num_frames).permute(0, 2, 1, 3).contiguous()

        sb_mask = sb_mask[:, :, :, self.look_ahead:]
        if self.decompress_mask:
            sb_mask = decompress_cIRM(sb_mask)

        enhanced_real = sb_mask[:, 0, :, :] * noisy_complex[:, 0, :, :] - sb_mask[:, 1, :, :] * noisy_complex[:, 1, :, :]
        enhanced_imag = sb_mask[:, 1, :, :] * noisy_complex[:, 0, :, :] + sb_mask[:, 0, :, :] * noisy_complex[:, 1, :, :]
        enhanced_complex = torch.stack((enhanced_real, enhanced_imag), dim=-1)

        estimated_samples = torch.istft(enhanced_complex, 512, 256)

        return estimated_samples, sb_mask


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

        enhanced_tensor, mask = model(ipt)
        enhanced_tensor, mask = enhanced_tensor.detach(), mask.permute(0,2,3,1).detach()
        print(enhanced_tensor.shape, mask.shape)
        print(f"Model Inference: {datetime.datetime.now() - start}")

        enhanced = torch.istft(mask, 512, 256, length=ipt_len)
        print(f"iSTFT: {datetime.datetime.now() - start}")

        print(f"{datetime.datetime.now() - start}")
        ipt = torch.rand(3, 800)
        enhanced_tensor, mask = model(ipt)
        print(enhanced_tensor.shape, mask.shape)
