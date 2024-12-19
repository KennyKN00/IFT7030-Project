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
# import fine_tuning as train
from synthesizer import train
from synthesizer.hparams import hparams

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

    mel_spectrogram = synthesizer.synthesize_spectrograms([text], embedding)

    return torch.tensor(mel_spectrogram)

def compute_initial_mel(audio_path, trained_model_path):
    synthesizer = Synthesizer(trained_model_path)
    mel = synthesizer.make_spectrogram(audio_path)

    plot_mel_spectrogram(mel, path="images/initial_mel.png")
    

def main():
    # recorder = Recorder()
    # recorder.run()
    
    # preprocessing.prepare_data(encoder, "audio_records/audio_val/metadata.csv")
    
    audio_path = "audio_records/reference.wav"
    pretrained_model_path = Path("models/synthesizer.pt")
    # fine_tuned_model_path = Path("models/checkpoint_epoch_10.pt")

    # Inital mel
    # compute_initial_mel(audio_path, pretrained_model_path)
    
    # Get mel before fine-tuning
    mel = clone_voice(audio_path, pretrained_model_path)
    # print(mel)
    # plot_mel_spectrogram(mel, path="images/before_fine_tuning_mel.png")

    # Get mel after fine-tuning
    # mel = clone_voice(audio_path, fine_tuned_model_path)
    # vocoder.writeMEL(mel, "Mels/TestAfterfinetuning.npy")
    # plot_mel_spectrogram(mel.squeeze(0), path="images/after_fine_tuning_mel.png")
    # mel_loaded = vocoder.readMEL("Mels/TestAfterfinetuning.npy")
    # print(mel_loaded.shape)

    # Generate a wav file from MEL
    wav_path = "outputs/example.wav"
    vocoder.generateWAV(mel, wav_path)
    print("Done!")


    # print(torch.load("models/checkpoint_epoch_10.pt"))

    train.train(run_id="",
      syn_dir="data/train/metadata.csv",
      models_dir=Path("models"),
      save_every=1000,
      backup_every=5000,
      force_restart=False,
      hparams=hparams,
      unfreeze_schedule={
          "encoder": 1e-5,  # Encoder unfreezes at epoch 1 with low LR
          "decoder": 1e-4,  # Decoder unfreezes at epoch 2 with higher LR
          "postnet": 1e-4   # Postnet unfreezes at epoch 3 with the same LR as decoder
      })


if __name__ == '__main__':
    main()