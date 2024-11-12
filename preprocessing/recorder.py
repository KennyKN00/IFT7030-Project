import time
import torch
import pandas as pd
import sounddevice as sd 
from scipy.io.wavfile import write
import os

MEL_SPECTROGRAM_DIR = "./mel_spectrograms/"
AUDIO_RECORDS_DIR = "./audio_records/"
CHAR_TO_ID = {c: i for i, c in enumerate("abcdefghijklmnopqrstuvwxyz' ")}

class Recorder:
    def __init__(self, encoder, inference, audio, sample_rate=16000, duration=5,):
        """
        Initialize the Recorder with paths, encoder, sample rate, and duration.

        Args:
            encoder: The encoder instance to generate embeddings.
            sample_rate (int): Sample rate for audio recording.
            duration (int): Duration in seconds for each recording.
        """
        self.encoder = encoder
        self.sample_rate = sample_rate
        self.duration = duration
        self.audio = audio
        self.inference = inference

        # Ensure directories exist
        os.makedirs(MEL_SPECTROGRAM_DIR, exist_ok=True)
        os.makedirs(AUDIO_RECORDS_DIR, exist_ok=True)

        # Load transcriptions
        self.transcriptions = pd.read_csv("data/transcriptions.csv")

    def record_audio(self):
        """Record audio for the set duration and sample rate."""
        print("\nRecording...")
        audio = sd.rec(int(self.duration * self.sample_rate), samplerate=self.sample_rate, channels=1, dtype='float32')
        sd.wait()  # Wait until recording is finished

        # Convert to torch tensor
        audio_tensor = torch.tensor(audio).squeeze()

        # Save the audio file
        write(os.path.join(AUDIO_RECORDS_DIR, "recording0.wav"), self.sample_rate, audio_tensor.numpy())

        return audio_tensor

    def preprocess_and_save(self, audio, filename, transcription):
        """
        Preprocess the audio, generate the mel-spectrogram and embedding, and save to disk.

        Args:
            audio (torch.Tensor): The recorded audio tensor.
            filename (str): The base filename to save the data.
            transcription (str): Transcription text for the audio.
        """
        # Convert audio tensor to NumPy array
        audio_np = audio.numpy()

        # Preprocess the audio using the encoder's audio module
        preprocessed_wav = self.audio.preprocess_wav(audio_np, self.sample_rate)

        # Compute the mel-spectrogram
        mel_spectrogram = self.audio.wav_to_mel_spectrogram(preprocessed_wav)

        # Generate the speaker embedding
        embedding = self.inference.embed_utterance(preprocessed_wav)

        # Convert mel-spectrogram and embedding to tensors
        mel_spectrogram_tensor = torch.from_numpy(mel_spectrogram)
        embedding_tensor = torch.from_numpy(embedding)

        text_sequence = text_to_sequence(transcription)
        
        # Save data
        data = {
            "mel_spectrogram": mel_spectrogram_tensor,
            "transcription": text_sequence,
            "speaker_embedding": embedding_tensor
        }
        torch.save(data, os.path.join(MEL_SPECTROGRAM_DIR, f"{filename}.pt"))
        print(f"Processed and saved: {filename}")

    def countdown(self, seconds):
        """Show a countdown timer."""
        for remaining in range(seconds, 0, -1):
            print(f"Starting in {remaining}...", end="\r")
            time.sleep(1)
        print(" " * 20, end="\r")  # Clear the line

    def run(self):
        """Run the recorder for all transcriptions in the CSV file."""
        user_input = input("Are you ready to start? (Y/N): ").strip().upper()
        if user_input != 'Y':
            print("Exiting...")
            return

        for _, row in self.transcriptions.iterrows():
            filename = row['filename'].split(".")[0]  # Strip the .wav extension for naming
            sentence = row['text']
            print(f"\nPlease read the sentence: '{sentence}'")

            # Start countdown timer
            self.countdown(3)

            # Record audio after countdown
            audio = self.record_audio()
            self.preprocess_and_save(audio, filename, sentence)

            # Ask the user if they want to continue
            user_input = input("Do you want to record the next sentence? (Y/N): ").strip().upper()
            if user_input != 'Y':
                print("Exiting...")
                break
    
def text_to_sequence(text, char_to_id=CHAR_TO_ID):
        """
        Convert a string of text into a sequence of integer IDs.
        
        :param text: str, the input transcription text
        :param char_to_id: dict, a dictionary mapping characters to IDs
        :return: torch.Tensor, a tensor containing the sequence of IDs
        """
        # Convert text to lowercase and map each character to its ID
        sequence = [char_to_id[char] for char in text.lower() if char in char_to_id]
        
        # Convert to a tensor and add a batch dimension
        return torch.tensor(sequence).unsqueeze(0)  # Shape: [1, sequence_length]