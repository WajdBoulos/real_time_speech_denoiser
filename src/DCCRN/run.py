import numpy as np

# from src.conv_tasnet import
from preprocess import preprocess
from train import train
import torch
from data import AudioDataset, AudioDataLoader

# Trying to imitate the run.sh script from the original github

# I'm using the Interspeech 2020 DNS Challenge Dataset. Dataset is put in egs/SE_dataset/tr and is organized as
# folders mix/clean/noise.

# To open visdom (shows loss graphs), run this command: "python -m visdom.server", and then open in browser
# http://localhost:8097

data_dir = "/home/saba-junior@staff.technion.ac.il/"
json_dir = "/home/saba-junior@staff.technion.ac.il/"

#train_dir = data_dir + "tr"
#valid_dir = data_dir + "cv"
#test_dir = data_dir + "tt"

id = 0
epochs = 100

# save and visualize

continue_from = ""
model_path = "DCCRN_sr_16k_batch_16_correct_BN_stft_lookaheadNEWWAJDNEW.pth"
model_features_path = ""  # i tried playing with a model for a deep feature loss, but it didn't work. So keep this empty

if __name__ == '__main__':
    sample_rate = 16000
    preprocess(data_dir, json_dir, sample_rate)

    batch_size = 16
    max_hours = None  # only use some of the data for tests. should be None when running on full dataset
    num_workers = 4
    train(data_dir, epochs, batch_size, model_path, model_features_path, max_hours=max_hours,
          continue_from=continue_from)
