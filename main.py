from pathlib import Path
import torchaudio
import torch
import encoder.inference as encoder
from preprocessing.recorder import Recorder
from vocoder import vocoder
import soundfile as sf
import evaluation.plot_mel as plot
import matplotlib.pyplot as plt
import librosa
import numpy as np
import preprocessing.data_preprocessing as preprocessing
from synthesizer.inference import Synthesizer
import fine_tuning as train

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

def clone_voice(audio_path, trained_path):
    encoder.load_model(Path("models/encoder.pt"))
    embedding = encoder.embed_utterance(encoder.preprocess_wav(audio_path))
    
    # Load the model
    synthesizer = Synthesizer(trained_path)

    # Input text
    text = ""
    
    text_path = "data/output_text.txt"
    with open(text_path, "r", encoding="utf-8") as file:
        text = file.read()

    # Generate mel spectrogram

    mel_spectrogram = synthesizer.synthesize_spectrograms([text], embedding)[0]

    return torch.tensor(mel_spectrogram)

def compute_initial_mel(audio_path, trained_model_path):
    synthesizer = Synthesizer(trained_model_path)
    mel = synthesizer.make_spectrogram(audio_path)

    plot_mel_spectrogram(mel, path="images/initial_mel.png")
    

def main():
    # recorder = Recorder()
    # recorder.run()
    
    # preprocessing.prepare_data(encoder, "audio_records/audio_val/metadata.csv")
    
    audio_path = "audio_records/reference4.wav"
    pretrained_model_path = Path("models/synthesizer.pt")
    fine_tuned_model_path = "models/fine_tuned_synthesizer.pt"
    
    # Inital mel
    # compute_initial_mel(audio_path, pretrained_model_path)
    
    # Get mel before fine-tuning
    mel = clone_voice(audio_path, pretrained_model_path)
    print(mel.shape)
    path = "MELTEST/mel8.npy"
    vocoder.writeMEL(mel, path)
    mel_read = vocoder.readMEL(path)
    print(mel_read.shape)
    # plot_mel_spectrogram(mel, path="images/before_fine_tuning_mel.png")
    # train.run()

    

if __name__ == '__main__':
    main()