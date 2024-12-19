import librosa
import mcd as mcd
import wer as wer

REF_AUDIO_PATH = "Presentation/reference5.wav"
SYNTH_AUDIO_PATH = "Presentation/Eval2.wav"
REF_TEXT_PATH = "Presentation/output_text3.txt"

ref_audio, sr = librosa.load(REF_AUDIO_PATH, sr=22050)
synth_audio, sr = librosa.load(SYNTH_AUDIO_PATH, sr=22050)

# Compute MCD
mcd_score = mcd.calculate_mcd(ref_audio, synth_audio, sr)

# Compute WER
wer_score = wer.compute_wer(REF_TEXT_PATH, SYNTH_AUDIO_PATH)

print("MCD SCORE: ", mcd_score, "WER SCORE:", wer_score)

def main():
    # Load audio
    ref_audio, sr = librosa.load(REF_AUDIO_PATH, sr=22050)
    synth_audio, sr = librosa.load(SYNTH_AUDIO_PATH, sr=22050)

    # Compute MCD
    mcd_score = mcd.calculate_mcd(ref_audio, synth_audio, sr)

    # Compute WER
    wer_score = wer.compute_wer(REF_TEXT_PATH, SYNTH_AUDIO_PATH)


    return mcd_score, wer_score
