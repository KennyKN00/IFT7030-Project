import torchaudio
import torch
from preprocessing.recorder import Recorder
from speechbrain.inference.TTS import MSTacotron2
from speechbrain.inference.vocoders import HIFIGAN
from vocoder import vocoder
import soundfile as sf

def main():
    # recorder = Recorder()
    # recorder.run()
    
    # Load the fine-tune model
    hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-libritts-22050Hz", savedir="pretrained_models/tts-hifigan-libritts-22050Hz")
    ms_tacotron2 = MSTacotron2.from_hparams(source="speechbrain/tts-mstacotron2-libritts", savedir="pretrained_models/tts-mstacotron2-libritts")
    fine_tuned_checkpoint = torch.load("models/fine_tuned_ms_tacotron2_epoch_5.pth", map_location=torch.device("cpu"))

    # Update the model's weights
    ms_tacotron2.mods['model'].load_state_dict(fine_tuned_checkpoint, strict=False)
    
    recording_path = "audio_records/reference.wav"
    output_text = ""
    
    output_text_path = "data/output_text.txt"
    with open(output_text_path, "r", encoding="utf-8") as file:
        output_text = file.read()
        
    mel_outputs, mel_lengths, alignments = ms_tacotron2.clone_voice(output_text, recording_path)
    
    #### Vocoder part
    # save the synthezied audio to audio_recordings/synthesized.wav
    y_hat = vocoder.librosaMEL2WAV(mel_outputs)
    sf.write("OutputInv.wav", y_hat, sr=22050)
    
    

if __name__ == '__main__':
    main()