import scipy.io.wavfile
import glob
from pathlib import Path
import os
import scipy.signal
import numpy as np

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
    totScoreStoi = 0;

    for iWav in range(0,len(clean_path_list)):
        ref_path = clean_path_list[iWav]
        deg_path = noisy_path_list[iWav]
        sample_rate1, ref = scipy.io.wavfile.read(ref_path)
        sample_rate2, deg = scipy.io.wavfile.read(deg_path)

        totScorePesq = totScorePesq + pesq(ref=np.array(ref), deg=np.array(deg), fs=sample_rate1, mode='nb')
        totScoreStoi = totScoreStoi + stoi(np.array(ref), np.array(deg), sample_rate2, extended=False)

    print("Pesq score is:" + str((totScorePesq)/float(len(clean_path_list))))
    print("Stoi score is :" + str((totScoreStoi)/float(len(clean_path_list))))

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

