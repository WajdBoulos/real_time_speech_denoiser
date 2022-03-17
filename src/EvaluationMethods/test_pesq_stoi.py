import scipy.io.wavfile
import glob
from pathlib import Path
import os
import scipy.signal
import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as signal
from matplotlib import pyplot as plt
import librosa
import librosa.display
import matplotlib.pyplot as plot


#pesq and stoi
from pesq import pesq
from pystoi import stoi

#for the mixing part only
from pydub import AudioSegment

#perform pesq and stoi test
def test_pesq_stoi():
    data_dir_clean = Path(__file__).parent.parent / 'Clean1'
    data_dir_noisy = Path(__file__).parent.parent / 'denoised_subnet_pretrained'

    clean_path_list = glob.glob(os.path.join(data_dir_clean, "*.wav"))
    noisy_path_list = glob.glob(os.path.join(data_dir_noisy, "*.wav"))
    totScorePesq = 0
    totScoreStoi = 0

    for iWav in range(0,len(clean_path_list)):
        ref_path = clean_path_list[iWav]
        deg_path = noisy_path_list[iWav]
        sample_rate1, ref = scipy.io.wavfile.read(ref_path)
        sample_rate2, deg = scipy.io.wavfile.read(deg_path)

        totScorePesq = totScorePesq + pesq(ref=np.array(ref), deg=np.array(deg), fs=sample_rate1, mode='nb')
        totScoreStoi = totScoreStoi + stoi(np.array(ref), np.array(deg), sample_rate2, extended=False)

    print("Pesq score is:" + str((totScorePesq)/float(len(clean_path_list))))
    print("Stoi score is :" + str((totScoreStoi)/float(len(clean_path_list))))

def plot_spectrogram():
    data_dir_clean = Path(__file__).parent / 'Clean'
    data_dir_noise = Path(__file__).parent / 'Noise'
    data_dir_noisy = Path(__file__).parent / 'Mix'
    data_dir_denoised_dccrn = Path(__file__).parent / 'denoised_audio_dccrn'
    data_dir_denoised_subnet = Path(__file__).parent / 'denoised_subnet'
    data_dir_denoised_dccrn_pretrained = Path(__file__).parent / 'denoised_audio_dccrn_pretrained'
    data_dir_denoised_subnet_pretrained = Path(__file__).parent / 'denoised_subnet_pretrained'

    clean_path_list = glob.glob(os.path.join(data_dir_clean, "*.wav"))
    noisy_path_list = glob.glob(os.path.join(data_dir_noisy, "*.wav"))
    noise_path_list = glob.glob(os.path.join(data_dir_noise, "*.wav"))
    denoised_dccrn_path_list = glob.glob(os.path.join(data_dir_denoised_dccrn, "*.wav"))
    noisy_subnet_path_list = glob.glob(os.path.join(data_dir_denoised_subnet, "*.wav"))
    denoised_dccrn_path_list_pretrained = glob.glob(os.path.join(data_dir_denoised_dccrn_pretrained, "*.wav"))
    noisy_subnet_path_list_pretrained = glob.glob(os.path.join(data_dir_denoised_subnet_pretrained, "*.wav"))


    for iWav in range(0,len(clean_path_list)):
        ref_path = clean_path_list[iWav]
        deg_path = noisy_path_list[iWav]
        noise_path = noise_path_list[iWav]
        denoised_dccrn_path = denoised_dccrn_path_list[iWav]
        denoised_subnet_path = noisy_subnet_path_list[iWav]
        denoised_dccrn_path_pretrained = denoised_dccrn_path_list_pretrained[iWav]
        denoised_subnet_path_pretrained = noisy_subnet_path_list_pretrained[iWav]

        sample_rate1, ref = scipy.io.wavfile.read(ref_path)
        sample_rate1, noise = scipy.io.wavfile.read(noise_path)
        sample_rate2, deg = scipy.io.wavfile.read(deg_path)
        sample_rate2, dccrn = scipy.io.wavfile.read(denoised_dccrn_path)
        sample_rate2, subnet = scipy.io.wavfile.read(denoised_subnet_path)
        sample_rate2, dccrn_pretrained = scipy.io.wavfile.read(denoised_dccrn_path_pretrained)
        sample_rate2, subnet_pretrained = scipy.io.wavfile.read(denoised_subnet_path_pretrained)

        fig = plt.figure()
        plot.subplot(711)
        powerSpectrum, freqenciesFound, time, imageAxis = plot.specgram(noise, Fs=sample_rate1,cmap=plt.cm.ocean)
        #plot.xlabel('Time')
        plot.ylabel('Frequency')
        plt.title('Noise')
        plt.xticks([])  # Command for hiding x-axis

        plot.subplot(712)
        powerSpectrum, freqenciesFound, time, imageAxis = plot.specgram(deg, Fs=sample_rate1,cmap=plt.cm.ocean)
        #plot.xlabel('Time')
        plot.ylabel('Frequency')
        plt.title('Noisy Speech')
        plt.xticks([])  # Command for hiding x-axis

        plot.subplot(713)
        powerSpectrum, freqenciesFound, time, imageAxis = plot.specgram(ref, Fs=sample_rate1, cmap=plt.cm.ocean)
        #plot.xlabel('Time')
        plot.ylabel('Frequency')
        plt.title('Clean Speech')
        plt.xticks([])  # Command for hiding x-axis

        plot.subplot(714)
        powerSpectrum, freqenciesFound, time, imageAxis = plot.specgram(dccrn, Fs=sample_rate1,cmap=plt.cm.ocean)
        #plot.xlabel('Time')
        plot.ylabel('Frequency')
        plt.title('DCCRN Denoised')
        plt.xticks([])  # Command for hiding x-axis
        plot.subplot(715)

        powerSpectrum, freqenciesFound, time, imageAxis = plot.specgram(dccrn_pretrained, Fs=sample_rate1,cmap=plt.cm.ocean)
        #plot.xlabel('Time')
        plot.ylabel('Frequency')
        plt.title('DCCRN_Pretrained Denoised')
        plt.xticks([])  # Command for hiding x-axis

        plot.subplot(716)
        powerSpectrum, freqenciesFound, time, imageAxis = plot.specgram(subnet, Fs=sample_rate1,cmap=plt.cm.ocean)
        #plot.xlabel('Time')
        plot.ylabel('Frequency')
        plt.title('Subnet Denoised')
        plt.xticks([])  # Command for hiding x-axis

        plot.subplot(717)
        powerSpectrum, freqenciesFound, time, imageAxis = plot.specgram(subnet_pretrained, Fs=sample_rate1,cmap=plt.cm.ocean)
        #plot.xlabel('Time')
        plot.ylabel('Frequency')
        plt.title('Subnet_Pretrained Denoised')
        # plt.xticks([])  # Command for hiding x-axis

        plt.subplots_adjust(left=0.05, right=0.95, top=0.98, bottom=0.04)
        plot.show()
        continue

#rename all files in folder
def rename_all():
    data_dir_in = Path(__file__).parent.parent / 'noisy_audio'
    os.chdir(data_dir_in)
    for filename in os.listdir(data_dir_in):
        os.rename(filename, "synthetic_pdns_noisy_fileid_" + str((filename.split("_")[-1])))

#mix clean wavs with noise wavs to produce noisy wavs
def mix():
    data_dir_clean = Path(__file__).parent.parent / 'Clean'
    data_dir_noise = Path(__file__).parent.parent / 'Noise'
    data_dir_mix = Path(__file__).parent.parent / 'Mix'

    clean_path_list = glob.glob(os.path.join(data_dir_clean, "*.wav"))
    noisy_path_list = glob.glob(os.path.join(data_dir_noise, "*.wav"))

    for iWav in range(0,len(clean_path_list)):

        ref_path = clean_path_list[iWav]
        deg_path = noisy_path_list[iWav]
        sound1 = AudioSegment.from_file(ref_path)
        sound2 = AudioSegment.from_file(deg_path)
        combined = sound1.overlay(sound2)

        file_name = os.path.basename(ref_path)
        id = (file_name.split('.'))[0].split('_')[2]

        combined.export((os.path.join(data_dir_mix, "out" + str(id) + ".wav")), format='wav')



if __name__=="__main__":
    #rename_all()
    #mix()
    test_pesq_stoi()
    #plot_spectrogram()


