import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from synthesizer.utils.text import text_to_sequence
import pandas as pd


class SynthesizerDataset(Dataset):
    def __init__(self, metadata_csv_path: Path, hparams):
        # Load the CSV file
        self.metadata = pd.read_csv(metadata_csv_path)
        
        # Validate required columns
        required_columns = ["audio_path", "transcript", "speaker_id", "speaker_embedding_path", "mel_path"]
        if not all(col in self.metadata.columns for col in required_columns):
            raise ValueError(f"Metadata file must contain the following columns: {required_columns}")
        
        # Extract file paths and texts
        self.mel_fpaths = self.metadata["mel_path"].map(Path).to_list()
        self.embed_fpaths = self.metadata["speaker_embedding_path"].map(Path).to_list()
        self.texts = self.metadata["transcript"].to_list()
        self.hparams = hparams

        print(f"Found {len(self.mel_fpaths)} samples in {metadata_csv_path}")
    
    def __getitem__(self, index):  
        # Load the mel spectrogram (as a PyTorch tensor)
        mel_path = self.mel_fpaths[index]
        mel = torch.tensor(torch.load(mel_path)).float()

        # Load the speaker embedding (as a PyTorch tensor)
        embed_path = self.embed_fpaths[index]
        embed = torch.tensor(torch.load(embed_path)).float()

        # Convert the text to a sequence of tokens
        text = text_to_sequence(self.texts[index], self.hparams.tts_cleaner_names)
        text = torch.tensor(text, dtype=torch.int32)

        return text, mel, embed, index

    def __len__(self):
        return len(self.mel_fpaths)


def collate_synthesizer(batch, r, hparams):
    # Text
    x_lens = [len(x[0]) for x in batch]
    max_x_len = max(x_lens)

    chars = [pad1d(x[0], max_x_len) for x in batch]
    chars = np.stack(chars)

    # Mel spectrogram
    spec_lens = [x[1].shape[-1] for x in batch]
    max_spec_len = max(spec_lens) + 1 
    if max_spec_len % r != 0:
        max_spec_len += r - max_spec_len % r 

    # WaveRNN mel spectrograms are normalized to [0, 1] so zero padding adds silence
    # By default, SV2TTS uses symmetric mels, where -1*max_abs_value is silence.
    if hparams.symmetric_mels:
        mel_pad_value = -1 * hparams.max_abs_value
    else:
        mel_pad_value = 0

    mel = [pad2d(x[1], max_spec_len, pad_value=mel_pad_value) for x in batch]
    mel = torch.stack(mel)

    # Speaker embedding (SV2TTS)
    embeds = torch.stack([x[2] for x in batch])

    # Index (for vocoder preprocessing)
    indices = [x[3] for x in batch]

    return chars, mel, embeds, indices

def pad1d(x, max_len, pad_value=0):
    return torch.nn.functional.pad(x, (0, max_len - len(x)), value=pad_value)

def pad2d(x, max_len, pad_value=0):
    return torch.nn.functional.pad(x, (0, max_len - x.shape[-1]), value=pad_value)
