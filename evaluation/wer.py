from speechbrain.inference import EncoderDecoderASR
from jiwer import wer
import torchaudio

def transform_to_text(audio_path):
    # Load ASR model
    asr_model = EncoderDecoderASR.from_hparams(
        source="speechbrain/asr-transformer-transformerlm-librispeech",
        savedir="pretrained_models/asr-transformer-transformerlm-librispeech"
    )

    # Load and preprocess audio
    audio_file = audio_path
    signal, sample_rate = torchaudio.load(audio_file)

    # Resample if necessary
    if sample_rate != 16000:
        signal = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(signal)

    # Perform transcription
    transcription = asr_model.transcribe_file(audio_file)
    
    return transcription

def compute_wer(reference_path, syntehsized_path):
    reference_text = ""
    with open(reference_path, "r", encoding="utf-8") as file:
        reference_text = file.read()
        
    synthesized_text = transform_to_text(syntehsized_path)

    synthesized_text = synthesized_text.lower()
    reference_text = reference_text.lower()
        
    return wer(reference_text, synthesized_text)
