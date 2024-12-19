import numpy as np
import librosa
import matplotlib.pyplot as plt
import soundfile as sf
import sounddevice as sd
import torch
import torchaudio

from speechbrain.inference.vocoders import HIFIGAN
from speechbrain.lobes.models.FastSpeech2 import mel_spectogram
from scipy.io.wavfile import write

hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-ljspeech",
                                savedir="pretrained_models/tts-hifigan-ljspeech")
hifi_gan_multi = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-libritts-22050Hz",
                                      savedir="pretrained_models/tts-hifigan-libritts-22050Hz")


def writeWAV(path, WAV, sr=22050):
    write(path, 22050, WAV)


def writeMEL(MEL, path):
    mel_np = MEL.numpy()
    np.save(path, mel_np)


def readMEL(path):
    mel_np = np.load(path)
    mel = torch.from_numpy(mel_np)
    return mel


def librosaWAV2MEL(path):  # From a file path
    y, sr = librosa.load(path)
    MEL = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=256, win_length=1024, n_mels=80)
    MEL = torch.from_numpy(MEL).to(device='cuda', dtype=torch.float32)
    return MEL


def HIFIWAV2MEL(path):  # From a file path
    # signal, rate = torchaudio.load(path)
    # signal, rate = torchaudio.load('speechbrain/tts-hifigan-ljspeech/example.wav')

    signal, sr = librosa.load(path)
    signal = torch.tensor(signal)
    spectrogram, _ = mel_spectogram(
        audio=signal.squeeze(),
        sample_rate=22050,
        hop_length=256,
        win_length=None,
        n_mels=80,
        n_fft=1024,
        f_min=0.0,
        f_max=8000.0,
        power=1,
        normalized=False,
        min_max_energy_norm=True,
        norm="slaney",
        mel_scale="slaney",
        compression=True
    )

    return spectrogram


def griffinLimMEL2WAV(MEL, sr=22050, n=1024):
    if isinstance(MEL, torch.Tensor):
        MEL = MEL.detach().cpu().numpy()
        MEL = MEL.astype(np.float32)
    S = librosa.feature.inverse.mel_to_stft(MEL, sr=sr, n_fft=1024)
    y_hat = librosa.istft(S, hop_length=256, win_length=n)
    y_hat = y_hat.flatten()
    return y_hat


def boost(audio, gain_db):
    gain = 10 ** (gain_db / 20)
    audio_boost = audio * gain

    return audio_boost


def HIFIGANMEL2WAV(MEL):
    y_hat = hifi_gan.decode_batch(MEL)
    y_hat = y_hat.flatten().numpy()
    return y_hat


def HIFIGAN_multi_MEL2WAV(MEL):
    y_hat = hifi_gan_multi.decode_batch(MEL)
    y_hat = y_hat.flatten().numpy()
    return y_hat

def generateWAV(MEL, path = "outputs/new_output.wav", P = True):
    if P:
        y_hat = HIFIGAN_multi_MEL2WAV(2* MEL) # Adapt to POWER MEL
    else:
        y_hat = HIFIGAN_multi_MEL2WAV(MEL)
    writeWAV(path, y_hat)
    return y_hat

def playWav(path):
    data, samplerate = sf.read(path)
    sd.play(data, samplerate)
    sd.wait()
