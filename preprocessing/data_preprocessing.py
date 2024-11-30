import os
import torchaudio.transforms as transforms
import random
import pandas as pd
import shutil
import numpy as np
from pathlib import Path
import torch
import librosa

# Paths to VCTK folders
TXT_PATH = "VCTK_data/txt"
AUDIO_PATH = "VCTK_data/wav48_silence_trimmed"
SAVE_PATH = "data"
os.makedirs(SAVE_PATH, exist_ok=True)

# Parameters for mel-spectrogram generation
SAMPLE_RATE = 16000
N_MELS = 80
HOP_LENGTH = 256
N_FFT = 1024

def normalize_path(path):
    return path.replace("\\", "/")

# Process each speaker's folder
def preparing():
    data = []
    
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
        
        # Select a few samples for the speaker (3 recordings)
        selected_samples = random.sample(audio_text_pairs, min(len(audio_text_pairs), 3))
        
        
        
        # Process selected samples
        for audio_path, transcript in selected_samples:
            # Define speaker-specific save path
            speaker_save_path = normalize_path(os.path.join(SAVE_PATH + "/preprocessed_data_test", speaker_id))
            os.makedirs(speaker_save_path, exist_ok=True)

            # Copy the audio file to the new location
            audio_save_path = normalize_path(os.path.join(speaker_save_path, os.path.basename(audio_path)))
            shutil.copy(audio_path, audio_save_path)

            # Add metadata
            data.append({"audio_path": audio_save_path, "transcript": transcript, "speaker_id": speaker_id})
            
    # Save metadata
    metadata_path = normalize_path(os.path.join(SAVE_PATH, "metadata.csv"))
    pd.DataFrame(data).to_csv(metadata_path, index=False)

    print(f"SUCCES : Preprocessing complete. Metadata saved to {metadata_path}.")


def processing(encoder, metadata_path):
    print("INFO : Starting the data's embeddings")
    encoder.load_model(Path("models/encoder.pt"))
    metadata = pd.read_csv(metadata_path)
    
    # Create a directory for saving embeddings
    embedding_folder = "data/val/speaker_embeddings"
    os.makedirs(embedding_folder, exist_ok=True)

    # Iterate through each row in metadata
    for idx, row in metadata.iterrows():
        audio_path = row["audio_path"]
        speaker_id = row["speaker_id"]

        # Generate speaker embedding
        embedding = encoder.embed_utterance(encoder.preprocess_wav(audio_path))

        # Save the embedding in a subfolder for the speaker
        speaker_folder = normalize_path(os.path.join(embedding_folder, speaker_id))
        os.makedirs(speaker_folder, exist_ok=True)

        # Save the embedding as a .pt file
        embedding_file = normalize_path(os.path.join(speaker_folder, os.path.basename(audio_path).replace(".flac", "_em.pt")))
        torch.save(embedding, embedding_file)
        
        metadata.loc[idx, "speaker_embedding_path"] = str(embedding_file)
        
    metadata.to_csv(metadata_path, index=False)
    print(f"Updated metadata saved to {metadata_path}")
    print(f"SUCCES : Embeddings complete. Speakers embeddings saved to {embedding_folder}.")
    
def audio_to_mel(metadata_path):
    metadata = pd.read_csv(metadata_path)
    
    mel_dir = "data/val/mel_spectrograms"
    os.makedirs(mel_dir, exist_ok=True)
    
    for index, row in metadata.iterrows():
        audio_path = row["audio_path"]

        # Generate mel-spectrogram
        mel_spec = compute_mel_spectrogram(audio_path)

        mel_folder = normalize_path(os.path.join(mel_dir, row["speaker_id"]))
        os.makedirs(mel_folder, exist_ok=True)
        
        # Save mel-spectrogram to file
        mel_path = normalize_path(os.path.join(mel_folder, os.path.basename(audio_path).replace(".flac", "_mel.pt")))
        torch.save(mel_spec, mel_path)

        # Add the mel_path to the metadata
        metadata.loc[index, "mel_path"] = str(mel_path)

    # Save updated metadata
    metadata.to_csv(metadata_path, index=False)
    print(f"Updated metadata saved to {metadata_path}")
    
def compute_mel_spectrogram(audio_path, sr=22050, n_fft=1024, hop_length=256, n_mels=80):
    # Load the audio
    y, sr = librosa.load(audio_path, sr=sr)
    
    # Compute the mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )
    
    # Convert to decibels
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    
    return mel_spectrogram_db
        
def prepare_data(encoder_inference, metadata_path):
    # preparing()
    processing(encoder_inference, metadata_path)
    audio_to_mel(metadata_path)
    