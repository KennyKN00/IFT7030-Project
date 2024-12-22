import librosa
import mcd as mcd
import wer as wer

REF_AUDIO_PATH = "audio_records/reference.wav"
SYNTH_AUDIO_PATH = "audio_records/synthesized.wav"
REF_TEXT_PATH = "data/output_text.txt"

def main():
    # Load audio
    ref_audio, sr = librosa.load(REF_AUDIO_PATH, sr=22050)
    synth_audio, sr = librosa.load(SYNTH_AUDIO_PATH, sr=22050)

    # Compute MCD
    mcd_score = mcd.calculate_mcd(ref_audio, synth_audio, sr)

    # Compute WER
    wer_score = wer.compute_wer(REF_TEXT_PATH, SYNTH_AUDIO_PATH)

    print(mcd_score, wer_score)

    return mcd_score, wer_score

main()
