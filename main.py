import torch
from preprocessing.recorder import Recorder
from speechbrain.inference.TTS import MSTacotron2

def main():
    recorder = Recorder()
    recorder.run()
    
    # Load the fine-tune model
    # ms_tacotron2 = MSTacotron2.from_hparams(source="models/fine_tuned_ms_tacotron2", savedir="pretrained_models")
    # recording_path = "audio_records/recording0.wav"
    # output_text = ""
    
    # output_text_path = "data/output_text.txt"
    # with open(output_text_path, "r", encoding="utf-8") as file:
    #     output_text = file.read()
        
    # mel_outputs, mel_lengths, alignments = ms_tacotron2.clone_voice(output_text, recording_path)
    
    #### Vocoder part
    
    

if __name__ == '__main__':
    main()