import numpy as np
import librosa
import matplotlib.pyplot as plt
import soundfile as sf
import torch


def loadSignal(filename):
    y, sr = librosa.load(filename)
    return y, sr

def librosaWAV2MEL(y, sr = 22050, n=2048):
    MEL = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n)
    return MEL

def librosaMEL2WAV(MEL, sr = 22050, n=1024):
    if isinstance(MEL, torch.Tensor):
        MEL = MEL.detach().cpu().numpy()
        MEL = MEL.astype(np.float32)
    S = librosa.feature.inverse.mel_to_stft(MEL, sr=sr, n_fft=n)
    y_hat = librosa.istft(S, hop_length=256, win_length=n)
    return y_hat

def griffinLimMEL2WAV(MEL, sr = 22050, n=2048):
    S = librosa.feature.inverse.mel_to_stft(MEL, sr=sr, n_fft=n)
    y_hat = librosa.griffinlim(S, hop_length=n // 4, win_length=n, window="hamming")
    return y_hat

def generateOutput(filename, algo = "inv"):
    y, sr = librosa.load(filename)
    MEL = librosaWAV2MEL(y, sr)
    y_hat = None
    if algo == "griffin":
        y_hat = griffinLimMEL2WAV(MEL, sr)
        sf.write("OutputGriffin.wav", y_hat, sr)
    elif algo == "inv":
        y_hat = librosaMEL2WAV(MEL, sr)
        sf.write("OutputInv.wav", y_hat, sr)
    elif algo == "waveglow":
        pass
    return y_hat

# Test

# f = 'input.wav'
# generateOutput(f, "griffin")
# generateOutput(f, "inv")
