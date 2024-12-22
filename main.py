from pathlib import Path
import torchaudio
import torch
import encoder.inference as encoder
from preprocessing.recorder import Recorder
from vocoder import vocoder
import soundfile as sf
import evaluation.plot_mel as plot
import matplotlib.pyplot as plt
import numpy as np
import preprocessing.data_preprocessing as preprocessing
from synthesizer.inference import Synthesizer
import fine_tuning as fine_tune
from synthesizer import train
from synthesizer.hparams import hparams
import warnings

warnings.filterwarnings(
    "ignore",
    message="You are using `torch.load` with `weights_only=False`",
    category=FutureWarning
)

def plot_mel_spectrogram(mel_spectrogram, sr=22050, hop_length=256, title="Mel-Spectrogram", path=""):
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

    plot_mel_spectrogram(mel, path="images/test_2.png")

def output_audio(mel, wav_path) :
    vocoder.generateWAV(mel, wav_path)
    print("Done!")


def main():
    # recorder = Recorder()
    # recorder.run()
    
    # preprocessing.prepare_data(encoder, "data/train/metadata.csv")
    
    audio_path = "data/reference.wav"
    pretrained_model_path = Path("models/synthesizer.pt")
    fine_tuned_model_path = Path("models/checkpoint_epoch_870.pt")

    ### Fine-tuning 1
    # train.train(run_id="",
    #   syn_dir="data/train/metadata.csv",
    #   models_dir=Path("models"),
    #   save_every=1000,
    #   backup_every=5000,
    #   force_restart=False,
    #   hparams=hparams,
    #   unfreeze_schedule={
    #       "encoder": 1e-5,  
    #       "decoder": 1e-4,  
    #       "postnet": 1e-4   
    #   })

    ### Fine-tuning 2
    # fine_tune.run()

    # Inital mel
    # compute_initial_mel("data/train/audio_train/p286/p286_003_mic1.wav", pretrained_model_path)
    
    ### Get mel before fine-tuning
    # mel_tensor = clone_voice(audio_path, pretrained_model_path)
    # mel_tensor = mel_tensor[0, :, :]
    # print(mel_tensor.shape)
    # plot_mel_spectrogram(mel_tensor, path="images/before_fine_tuning_mel.png")
    # output_audio(mel_tensor, "outputs/synthesized.wav")

    ### Get mel after fine-tuning
    # mel_tensor = clone_voice(audio_path, fine_tuned_model_path)
    # mel_tensor = mel_tensor[0, :, :]
    # print(mel_tensor.shape)
    # plot_mel_spectrogram(mel_tensor, path="images/another_test_500_1.png")
    # output_audio(mel_tensor, "outputs/synthesized.wav")


if __name__ == '__main__':
    main()