import matplotlib.pyplot as plt
import numpy as np

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
