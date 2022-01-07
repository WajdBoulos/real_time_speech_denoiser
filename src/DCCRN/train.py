#!/usr/bin/env python

# Created on 2018/12
# Author: Kaituo XU


import torch
from data import AudioDataLoader, AudioDataset
from solver import Solver
from src.DCCRN.DCCRN import DCCRN
from torch.utils.data.dataset import random_split
import math
from fairseq.models.wav2vec import Wav2VecModel


def train(data_dir, epochs, batch_size, model_path, model_features_path, max_hours=None, continue_from=""):
    # General config
    # Task related
    train_dir = data_dir + "tr"
    sample_rate = 16000
    segment_len = 4

    # Network architecture
    fft_len = 512
    win_len = int(25e-3 * sample_rate)
    hop_size = int(6.25e-3 * sample_rate)
    window = 'hann'
    enc_list = [16, 32, 64, 64, 128, 128]
    dec_list = [x * 2 for x in enc_list]
    dec_list.reverse()
    num_convs = len(enc_list)
    freq_kernel_size = 5
    time_kernel_size = 2
    stride = (2, 1)
    dilation = 1
    norm_type = 'BN'
    rnn_type = 'LSTM'
    num_layers = 2
    mask_type = 'E'

    use_cuda = 1
    device = torch.device("cuda" if use_cuda else "cpu")
    half_lr = 1  # Half the learning rate when there's a small improvement
    early_stop = 1  # Stop learning if no imporvement after 10 epochs
    max_grad_norm = 5  # gradient clipping

    shuffle = 1  # Shuffle every epoch
    num_workers = 4
    # optimizer
    optimizer_type = "adam"
    lr = 5e-4
    momentum = 0
    l2 = 0  # Weight decay - l2 norm

    # save and visualize
    save_folder = "../egs/models"
    enable_checkpoint = 0  # enables saving checkpoints
    print_freq = 100
    visdom_enabled = 0
    visdom_epoch = 0
    visdom_id = "Conv-TasNet Training"  # TODO: Check what this does

    # deep_features_model = DeepSpeech.load_model(model_features_path)
    # deep_features_model.eval()

    # cp = torch.load('../egs/models/loss_models/wav2vec_large.pt')
    # deep_features_model = Wav2VecModel.build_model(cp['args'], task=None)
    # deep_features_model.load_state_dict(cp['model'])
    # deep_features_model.eval()
    deep_features_model = None

    arg_solver = (use_cuda, epochs, half_lr, early_stop, max_grad_norm, save_folder, enable_checkpoint, continue_from,
                  model_path, print_freq, visdom_enabled, visdom_epoch, visdom_id, deep_features_model, device)

    # Datasets and Dataloaders
    tr_cv_dataset = AudioDataset(train_dir, batch_size,
                                 sample_rate=sample_rate, segment=segment_len, max_hours=max_hours)
    cv_len = int(0.2 * math.ceil(len(tr_cv_dataset)))
    tr_dataset, cv_dataset = random_split(tr_cv_dataset, [len(tr_cv_dataset) - cv_len, cv_len])

    tr_loader = AudioDataLoader(tr_dataset, batch_size=1,
                                shuffle=shuffle,
                                num_workers=num_workers)
    cv_loader = AudioDataLoader(cv_dataset, batch_size=1,
                                num_workers=num_workers)
    data = {'tr_loader': tr_loader, 'cv_loader': cv_loader}

    #model = DCCRN_DS(fft_len, win_len, hop_size, window, num_convs, enc_list, dec_list, 
    model = DCCRN(fft_len, win_len, hop_size, window, num_convs, enc_list, dec_list,
freq_kernel_size,
                     time_kernel_size, stride, dilation, norm_type, rnn_type, num_layers, mask_type)

    if use_cuda:
        model = torch.nn.DataParallel(model)
        model.cuda()
    # optimizer
    if optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=lr,
                                    momentum=momentum,
                                    weight_decay=l2)
    elif optimizer_type == 'adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=lr,
                                     weight_decay=l2)
    else:
        print("Not support optimizer")
        return

    # solver
    solver = Solver(data, model, optimizer, arg_solver)
    solver.train()


if __name__ == '__main__':
    print('train main')
    # args = parser.parse_args()
    # print(args)
    # train(args)
