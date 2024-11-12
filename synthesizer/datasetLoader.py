import torch
from torch.utils.data import Dataset

class Tacotron2FineTuneDataset(Dataset):
    def __init__(self, pt_file_paths):
        self.pt_file_paths = pt_file_paths

    def __len__(self):
        return len(self.pt_file_paths)

    def __getitem__(self, idx):
        # Load data from .pt file
        data = torch.load(self.pt_file_paths[idx])
        text_ids = data["transcription"]  # Sequence of character IDs
        mel_spectrogram = data["mel_spectrogram"]  # Mel spectrogram
        speaker_embedding = data["speaker_embedding"]  # Speaker embedding
        
        return text_ids, mel_spectrogram, speaker_embedding
