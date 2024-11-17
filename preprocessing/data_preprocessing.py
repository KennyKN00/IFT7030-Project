import os
import torchaudio
import torchaudio.transforms as transforms
import torch
import random
import pandas as pd

# Paths to VCTK folders
TXT_PATH = "VCTK_data/txt"
AUDIO_PATH = "VCTK_data/wav48_silence_trimmed"
SAVE_PATH = "data/preprocessed_data"
os.makedirs(SAVE_PATH, exist_ok=True)

# Parameters for mel-spectrogram generation
SAMPLE_RATE = 22050
N_MELS = 80
HOP_LENGTH = 256
N_FFT = 1024

def normalize_path(path):
    return path.replace("\\", "/")

# Create mel-spectrogram transformation
mel_transform = transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS
)

# Prepare data list for episodic fine-tuning
data = []

# Process each speaker's folder
print("INFO : Starting the data's preprocessing ")
for speaker_id in os.listdir(TXT_PATH):
    speaker_txt_folder = os.path.join(TXT_PATH, speaker_id)
    speaker_audio_folder = os.path.join(AUDIO_PATH, speaker_id)
    
    speaker_txt_folder = normalize_path(speaker_txt_folder)
    speaker_audio_folder = normalize_path(speaker_audio_folder)
    
    
    if not os.path.isdir(speaker_txt_folder) or not os.path.isdir(speaker_audio_folder):
        continue

    # Get all audio-text pairs for the speaker
    audio_text_pairs = []
    for txt_file in os.listdir(speaker_txt_folder):
        if txt_file.endswith(".txt"):
            transcript_path = os.path.join(speaker_txt_folder, txt_file)
            audio_file = txt_file.replace(".txt", "_mic1.flac")
            audio_path = os.path.join(speaker_audio_folder, audio_file)
            transcript_path = normalize_path(transcript_path)
            audio_path = normalize_path(audio_path)
            
            if os.path.exists(audio_path):
                with open(transcript_path, "r") as f:
                    transcript = f.read().strip()
                audio_text_pairs.append((audio_path, transcript))
    
    # Select a few samples for the speaker (e.g., 2–3 recordings)
    selected_samples = random.sample(audio_text_pairs, min(len(audio_text_pairs), 3))
    
    # Process selected samples
    for audio_path, transcript in selected_samples:
        # Load audio
        waveform, sr = torchaudio.load(audio_path)

        # Resample if necessary
        if sr != SAMPLE_RATE:
            resampler = transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
            waveform = resampler(waveform)

        # Generate mel-spectrogram
        mel_spectrogram = mel_transform(waveform).squeeze(0)

        speaker_save_path = normalize_path(os.path.join(SAVE_PATH, speaker_id))
        os.makedirs(speaker_save_path, exist_ok=True)

        # Save the mel-spectrogram in the speaker's folder
        mel_save_path = normalize_path(os.path.join(speaker_save_path, f"{os.path.basename(audio_path).replace('.flac', '.pt')}"))
        torch.save(mel_spectrogram, mel_save_path)

        # Append to data list
        data.append({"mel_path": mel_save_path, "transcript": transcript, "speaker_id": speaker_id})

# Save metadata
metadata_path = normalize_path(os.path.join(SAVE_PATH, "metadata.csv"))
pd.DataFrame(data).to_csv(metadata_path, index=False)

print(f"SUCCES : Preprocessing complete. Metadata saved to {metadata_path}.")

