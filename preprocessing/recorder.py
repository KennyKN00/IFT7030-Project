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
    def __init__(self, sample_rate=16000, duration=10,):
        """
        Initialize the Recorder with paths, encoder, sample rate, and duration.

        Args:
            sample_rate (int): Sample rate for audio recording.
            duration (int): Duration in seconds for each recording.
        """
        self.sample_rate = sample_rate
        self.duration = duration

        # Ensure directories exist
        os.makedirs(MEL_SPECTROGRAM_DIR, exist_ok=True)
        os.makedirs(AUDIO_RECORDS_DIR, exist_ok=True)

        # Load transcriptions
        self.transcriptions = pd.read_csv("data/transcriptions.csv")

    def record_audio(self):
        """Record audio for the set duration and sample rate."""
        print("\nRecording...")
        audio = sd.rec(int(self.duration * self.sample_rate), samplerate=self.sample_rate, channels=1, dtype='float32')
        sd.wait()  
        
        # Convert to torch tensor
        audio_tensor = torch.tensor(audio).squeeze()

        # Save the audio file
        write(os.path.join(AUDIO_RECORDS_DIR, "reference.wav"), self.sample_rate, audio_tensor.numpy())

        return audio_tensor

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