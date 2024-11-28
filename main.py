import torchaudio
import torch
import encoder.audio
import encoder.inference
from preprocessing.recorder import Recorder
from speechbrain.inference.TTS import MSTacotron2
from speechbrain.inference.vocoders import HIFIGAN
from vocoder import vocoder
import soundfile as sf
import evaluation.plot_mel as plot
import matplotlib.pyplot as plt
import librosa
import numpy as np
import preprocessing.data_preprocessing as preprocessing

def main():
    # recorder = Recorder()
    # recorder.run()
    
    preprocessing.prepare_data(encoder.inference, encoder.audio)
    
    # # Load the fine-tune model
    # # hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-libritts-22050Hz", savedir="pretrained_models/tts-hifigan-libritts-22050Hz")
    # ms_tacotron2 = MSTacotron2.from_hparams(source="speechbrain/tts-mstacotron2-libritts", savedir="pretrained_models/tts-mstacotron2-libritts")
    # fine_tuned_checkpoint = torch.load("models/fine_tuned_ms_tacotron2_epoch_5.pth", map_location=torch.device("cpu"))
    
    # # Update the model's weights
    # # ms_tacotron2.mods['model'].load_state_dict(fine_tuned_checkpoint, strict=False)
    
    # recording_path = "audio_records/reference.wav"
    # output_text = ""
    
    # output_text_path = "data/output_text.txt"
    # with open(output_text_path, "r", encoding="utf-8") as file:
    #     output_text = file.read()
        
    # mel_outputs, mel_lengths, alignments = ms_tacotron2.clone_voice(output_text, recording_path)
    # plot(mel_outputs.squeeze(0))
    # #### Vocoder part
    # # save the synthezied audio to audio_recordings/synthesized.wav
    # # y_hat = vocoder.librosaMEL2WAV(mel_outputs)
    # # sf.write("OutputInv.wav", y_hat, sr=22050)
    
    # audio_path = "data/preprocessed_data_test/p225/p225_008_mic1.flac"
    # y, sr = librosa.load(audio_path, sr=22050)  # Load as waveform, resampling to 22.05 kHz

    # # Compute the mel-spectrogram
    # mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80, hop_length=256, win_length=1024)
    # mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)  # Convert to dB scale for visualization

    # # Plot the mel-spectrogram
    # plt.figure(figsize=(10, 4))
    # librosa.display.specshow(mel_spec_db, sr=sr, hop_length=256, x_axis="time", y_axis="mel", cmap="viridis")
    # plt.colorbar(format="%+2.0f dB")
    # plt.title("Mel-Spectrogram")
    # plt.tight_layout()
    # plt.show()
    
    

if __name__ == '__main__':
    main()