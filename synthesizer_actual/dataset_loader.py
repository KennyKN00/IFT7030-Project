import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    # Extract mel-spectrograms, transcripts, and speaker IDs
    mel_targets = [item["mel"].transpose(0, 1) for item in batch]  # Transpose to [time, frequency]
    transcripts = [item["transcript"] for item in batch]
    speaker_ids = [item["speaker_id"] for item in batch]


    # Pad mel-spectrograms along the time dimension
    mel_targets_padded = pad_sequence(mel_targets, batch_first=True, padding_value=0.0)  # [batch, max_time, frequency]
    mel_targets_padded = mel_targets_padded.transpose(1, 2)  # Back to [batch, frequency, max_time]

    return {
        "mel": mel_targets_padded,
        "transcript": transcripts,
        "speaker_id": speaker_ids,
    }

class FewShotVCTKDataset(Dataset):
    def __init__(self, metadata_path):
        self.metadata = pd.read_csv(metadata_path)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        mel = torch.load(row["mel_path"])
        transcript = row["transcript"]
        speaker_id = row["speaker_id"]
        return {"mel": mel, "transcript": transcript, "speaker_id": speaker_id}

# Create dataset and dataloader
dataset = FewShotVCTKDataset("data/preprocessed_data/metadata.csv")
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
