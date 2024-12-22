from vocoder import vocoder
import matplotlib.pyplot as plt
import torch
import librosa

import os

def plot_mel_spectrogram(mel_spectrogram, sr=22050, hop_length=512, title="Mel-Spectrogram", path=""):
    plt.figure(figsize=(10, 4))
    plt.imshow(mel_spectrogram, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(format="%+2.0f dB")
    plt.title(title)
    plt.ylabel("Mel Frequency")
    plt.xlabel("Time (frames)")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.show()


def buildMELData():
    path = "data/train/preprocessed_data_test"
    save_path = "mel_spectrograms2"

    for r, d, all_f in os.walk(path):
        for f in all_f:
            p = os.path.join(r, f)
            MEL = vocoder.HIFIWAV2MEL(p)
            new_p = p.split("test")[1]
            rep = os.path.dirname(new_p)
            mk_dir = save_path + rep

            try:
                os.makedirs(mk_dir)
            except FileExistsError:
                pass

            mel_p = save_path + new_p
            mel_p = mel_p.split(".")[0]
            mel_p += "_mel.pt"
            torch.save(MEL, mel_p)

    print("Done!")
    return

path = "data/train/mel_spectrograms/p225/p225_349_mic1_mel.pt"
MEL = torch.load(path)
print(MEL.shape)
print(type(MEL))
plot_mel_spectrogram(MEL, path="images/trainlookmel6.png")

# path = "data/train/preprocessed_data_test/p225/p225_349_mic1.flac"
# MEL = vocoder.HIFIWAV2MEL(path)
# print(MEL.shape)
# plot_mel_spectrogram(MEL, path="images/trainlookmel2.png")
# sav = "finetuning/meltest.pt"
# torch.save(MEL, sav)
# MEL = torch.load(sav)
# print(MEL.shape)








