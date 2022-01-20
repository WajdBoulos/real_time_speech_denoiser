# Created on 2018/12
# Author: Kaituo XU

def decompress_cIRM(mask, K=10, limit=9.9):
    mask = limit * (mask >= limit) - limit * (mask <= -limit) + mask * (torch.abs(mask) < limit)
    mask = -K * torch.log((K - mask) / (K + mask))
    return mask

def compress_cIRM(mask, K=10, C=0.1):
    """
        Compress from (-inf, +inf) to [-K ~ K]
    """
    if torch.is_tensor(mask):
        mask = -100 * (mask <= -100) + mask * (mask > -100)
        mask = K * (1 - torch.exp(-C * mask)) / (1 + torch.exp(-C * mask))
    else:
        mask = -100 * (mask <= -100) + mask * (mask > -100)
        mask = K * (1 - np.exp(-C * mask)) / (1 + np.exp(-C * mask))
    return mask

import math
import numpy as np
import torch

NEG_INF = torch.finfo(torch.float32).min
PI = math.pi
SOUND_SPEED = 343  # m/s
EPSILON = np.finfo(np.float32).eps
MAX_INT16 = np.iinfo(np.int16).max

import os
import time

from torch.cuda.amp import autocast
from tqdm import tqdm
import visdom
import torch

from loss_functions import cal_loss

def build_complex_ideal_ratio_mask(noisy: torch.complex64, clean: torch.complex64) -> torch.Tensor:
    """

    Args:
        noisy: [B, F, T], noisy complex-valued stft coefficients
        clean: [B, F, T], clean complex-valued stft coefficients

    Returns:
        [B, F, T, 2]
    """
    denominator = torch.square(noisy.real) + torch.square(noisy.imag) + EPSILON

    mask_real = (noisy.real * clean.real + noisy.imag * clean.imag) / denominator
    mask_imag = (noisy.real * clean.imag - noisy.imag * clean.real) / denominator

    complex_ratio_mask = torch.stack((mask_real, mask_imag), dim=-1)

    return compress_cIRM(complex_ratio_mask, K=10, C=0.1)

def mag_phase(complex_tensor):
    return torch.abs(complex_tensor), torch.angle(complex_tensor)

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


class Solver(object):

    def __init__(self, data, model, optimizer, arg_solver):

        (use_cuda, epochs, half_lr, early_stop, max_grad_norm, save_folder, enable_checkpoint, continue_from,
         model_path, print_freq, visdom_enabled, visdom_epoch, visdom_id, deep_features_model, device) = arg_solver

        self.tr_loader = data['tr_loader']
        self.cv_loader = data['cv_loader']
        self.model = model
        self.deep_features_model = deep_features_model
        self.optimizer = optimizer
        self.device = device
        # Training config
        self.use_cuda = use_cuda
        self.epochs = epochs
        self.half_lr = half_lr
        self.early_stop = early_stop
        self.max_norm = max_grad_norm
        # save and load model
        self.save_folder = save_folder
        self.enable_checkpoint = enable_checkpoint
        self.continue_from = continue_from
        self.model_path = model_path
        # logging
        self.print_freq = print_freq
        # visualizing loss using visdom
        self.tr_loss = torch.Tensor(self.epochs)
        self.cv_loss = torch.Tensor(self.epochs)
        self.visdom_enabled = visdom_enabled
        self.visdom_epoch = visdom_epoch
        self.visdom_id = visdom_id
        if self.visdom_enabled:
            from visdom import Visdom
            self.vis = Visdom(env=self.visdom_id)
            self.vis_opts = dict(title=self.visdom_id,
                                 ylabel='Loss', xlabel='Epoch',
                                 legend=['train loss', 'cv loss'])
            self.vis_window = None
            self.vis_epochs = torch.arange(1, self.epochs + 1)

        self._reset()

    def _reset(self):
        # Reset
        if self.continue_from:
            print('Loading checkpoint model %s' % self.continue_from)
            file_path = os.path.join(self.save_folder, self.continue_from)
            package = torch.load(file_path)
            self.model.module.load_state_dict(package['state_dict'])
            self.optimizer.load_state_dict(package['optim_dict'])
            self.start_epoch = package['epoch']
            self.epochs = self.epochs + self.start_epoch
            self.tr_loss = torch.Tensor(self.epochs)
            self.cv_loss = torch.Tensor(self.epochs)
            self.vis_epochs = torch.arange(1, self.epochs + 1)
            self.epochs = self.epochs + self.start_epoch
            self.tr_loss[:self.start_epoch] = package['tr_loss'][:self.start_epoch]
            self.cv_loss[:self.start_epoch] = package['cv_loss'][:self.start_epoch]
        else:
            self.start_epoch = 0
        # Create save folder
        os.makedirs(self.save_folder, exist_ok=True)
        self.best_val_loss = float("inf")
        self.halving = False
        self.val_no_impv = 0

    def train(self):
        # Train model multi-epoches
        for epoch in (range(self.start_epoch, self.epochs)):
            # Train one epoch
            print("Training...")
            self.model.train()  # Turn on BatchNorm & Dropout
            start = time.time()
            tr_avg_loss = self._run_one_epoch(epoch, self.device)
            print('-' * 85)
            print('Train Summary | End of Epoch {0} | Time {1:.2f}s | '
                  'Train Loss {2:.3f}'.format(
                epoch + 1, time.time() - start, tr_avg_loss))
            print('-' * 85)

            # Save model each epoch
            if self.enable_checkpoint:
                file_path = os.path.join(self.save_folder,
                                         "checkpoint_models", 'epoch%d.pth.tar' % (epoch + 1))
                torch.save(self.model.module.serialize(self.model.module,
                                                       self.optimizer, epoch + 1,
                                                       tr_loss=self.tr_loss,
                                                       cv_loss=self.cv_loss),
                           file_path)
                print('Saving checkpoint model to %s' % file_path)

            # Cross validation
            print('Cross validation...')
            self.model.eval()  # Turn off Batchnorm & Dropout
            with torch.no_grad():
                val_loss = self._run_one_epoch(epoch, self.device, cross_valid=True)
            print('-' * 85)
            print('Valid Summary | End of Epoch {0} | Time {1:.2f}s | '
                  'Valid Loss {2:.3f}'.format(
                epoch + 1, time.time() - start, val_loss))
            print('-' * 85)

            # Adjust learning rate (halving)
            if self.half_lr:
                if val_loss >= self.best_val_loss:
                    self.val_no_impv += 1
                    if self.val_no_impv == 3:
                        self.halving = True
                    if self.val_no_impv >= 7 and self.early_stop:
                        print("No imporvement for 7 epochs, early stopping.")
                        break
                else:
                    self.val_no_impv = 0
            if self.halving:
                optim_state = self.optimizer.state_dict()
                optim_state['param_groups'][0]['lr'] = \
                    optim_state['param_groups'][0]['lr'] / 2.0
                self.optimizer.load_state_dict(optim_state)
                print('Learning rate adjusted to: {lr:.6f}'.format(
                    lr=optim_state['param_groups'][0]['lr']))
                self.halving = False

            # Save the best model
            self.tr_loss[epoch] = tr_avg_loss
            self.cv_loss[epoch] = val_loss
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                file_path = os.path.join(self.save_folder, self.model_path)
                torch.save(self.model.module.serialize(self.model.module,
                                                       self.optimizer, epoch + 1,
                                                       tr_loss=self.tr_loss,
                                                       cv_loss=self.cv_loss),
                           file_path)
                print("Found better validated model, saving to %s" % file_path)

            # visualizing loss using visdom
            if self.visdom_enabled:
                x_axis = self.vis_epochs[0:epoch + 1]
                y_axis = torch.stack(
                    (self.tr_loss[0:epoch + 1], self.cv_loss[0:epoch + 1]), dim=1)
                if self.vis_window is None:
                    self.vis_window = self.vis.line(
                        X=x_axis,
                        Y=y_axis,
                        opts=self.vis_opts,
                    )
                else:
                    self.vis.line(
                        X=x_axis.unsqueeze(0).expand(y_axis.size(
                            1), x_axis.size(0)).transpose(0, 1),  # Visdom fix
                        Y=y_axis,
                        win=self.vis_window,
                        update='replace',
                    )

    def _run_one_epoch(self, epoch, device, cross_valid=False):
        start = time.time()
        total_loss = 0

        data_loader = self.tr_loader if not cross_valid else self.cv_loader

        # visualizing loss using visdom
        if self.visdom_epoch and not cross_valid:
            vis_opts_epoch = dict(title=self.visdom_id + " epoch " + str(epoch),
                                  ylabel='Loss', xlabel='Epoch')
            vis_window_epoch = None
            vis_iters = torch.arange(1, len(data_loader) + 1)
            vis_iters_loss = torch.Tensor(len(data_loader))
        i = 0
        for (data_package) in tqdm(data_loader):
            padded_mixture, mixture_lengths, padded_clean_noise = data_package
            if self.use_cuda:
                padded_mixture = padded_mixture.cuda()
                mixture_lengths = mixture_lengths.cuda()
                padded_clean_noise = padded_clean_noise.cuda()

            noisy_complex = torch.stft((padded_mixture).to(device), n_fft=512, hop_length=256,
                                       window=torch.hann_window(512).to(device), return_complex=True)
            noisy_mag, _ = mag_phase(noisy_complex)
            noisy_mag = noisy_mag.unsqueeze(1)

            pred_crm = self.model(noisy_mag).permute(0, 2, 3, 1).to(device)

            # pr.disable()
            # pr.print_stats(sort='time')

            pred_crm = decompress_cIRM(pred_crm)
            enhanced_real = pred_crm[..., 0] * noisy_complex.real - pred_crm[..., 1] * noisy_complex.imag
            enhanced_imag = pred_crm[..., 1] * noisy_complex.real + pred_crm[..., 0] * noisy_complex.imag
            enhanced_complex = torch.stack((enhanced_real, enhanced_imag), dim=-1)

            estimate_source = torch.istft(enhanced_complex, 512, 256)

            source = padded_clean_noise[:, 0, :]  # first arg is source, second is noise
            loss = cal_loss(source, estimate_source, mixture_lengths, device, features_model=self.deep_features_model)
            if not cross_valid:
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               self.max_norm)
                self.optimizer.step()
            # import GPUtil
            total_loss += loss.item()

            # if i % self.print_freq == 0:
            #    print('Epoch {0} | Iter {1} | Average Loss {2:.3f} | '
            #          'Current Loss {3:.6f} | {4:.1f} ms/batch'.format(
            #        epoch + 1, i + 1, total_loss / (i + 1),
            #        loss.item(), 1000 * (time.time() - start) / (i + 1)),
            #        flush=True)

            # visualizing loss using visdom
            if self.visdom_epoch and not cross_valid:
                vis_iters_loss[i] = loss.item()
                if i % self.print_freq == 0:
                    x_axis = vis_iters[:i + 1]
                    y_axis = vis_iters_loss[:i + 1]
                    if vis_window_epoch is None:
                        vis_window_epoch = self.vis.line(X=x_axis, Y=y_axis,
                                                         opts=vis_opts_epoch)
                    else:
                        self.vis.line(X=x_axis, Y=y_axis, win=vis_window_epoch,
                                      update='replace')
            i += 1
        return total_loss / (i + 1)
