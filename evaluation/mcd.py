import librosa
import numpy as np

def calculate_mcd(ref_audio, synth_audio, sr=22050):
    ref_mfcc = librosa.feature.mfcc(y=ref_audio, sr=sr, n_mfcc=13)
    synth_mfcc = librosa.feature.mfcc(y=synth_audio, sr=sr, n_mfcc=13)
    min_frames = min(ref_mfcc.shape[1], synth_mfcc.shape[1])
    ref_mfcc = ref_mfcc[:, :min_frames]
    synth_mfcc = synth_mfcc[:, :min_frames]
    diff = ref_mfcc - synth_mfcc
    mcd = (10 / np.log(10)) * np.sqrt(2 * np.sum(diff**2, axis=0)).mean()
    
    return mcd