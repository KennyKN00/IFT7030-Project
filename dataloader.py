import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd

class TTSDataLoader(Dataset):
    def __init__(self, metadata_path, vocab):
        """
        Args:
            metadata_path (str): Path to the metadata.csv file.
            vocab (dict): Mapping of characters to integer indices.
        """
        self.metadata = pd.read_csv(metadata_path)
        self.vocab = vocab
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        """
        Retrieves a single sample from the dataset.
        Args:
            idx (int): Index of the sample.
        
        Returns:
            dict: Dictionary containing:
                - 'transcript_sequence': Tensor of the tokenized transcript.
                - 'mel': Tensor of the mel spectrogram.
                - 'embedding': Tensor of the speaker embedding.
                - 'stop_target': Tensor of stop token targets.
        """
        row = self.metadata.iloc[idx]
        transcript = row['transcript']
        mel_path = row['mel_path']
        embeddings_path = row['speaker_embedding_path']

        # Convert transcript to sequence
        text_sequence = [self.vocab[char] for char in transcript.lower() if char in self.vocab]
        text_sequence = torch.tensor(text_sequence)
        
        # Load mel spectrogram and embeddings
        mel = torch.tensor(torch.load(mel_path, weights_only=False))  # Assumes .pt files are PyTorch tensors
        embedding = torch.tensor(torch.load(embeddings_path, weights_only=False))  # Assumes .pt files are PyTorch tensors

        # Generate stop token target (1 at the last frame, 0 elsewhere)
        stop_target = torch.zeros(mel.size(1))
        stop_target[-1] = 1.0

        return {
            'transcript_sequence': text_sequence,
            'mel': mel,
            'embedding': embedding,
            'stop_target': stop_target
        }

def collate_fn(batch):
    """
    Custom collate function for padding variable-length inputs.
    Args:
        batch (list): List of samples (dictionaries).
    
    Returns:
        dict: Batched data with padded sequences.
    """
    transcripts = [item['transcript_sequence'] for item in batch]
    embeddings = torch.stack([item['embedding'] for item in batch])

    # Find max mel length
    mel_lengths = [item['mel'].size(1) for item in batch]
    max_mel_len = max(mel_lengths)

    # Pad mel spectrograms and stop targets
    mels = torch.zeros(len(batch), batch[0]['mel'].size(0), max_mel_len)
    stop_targets = torch.zeros(len(batch), max_mel_len)

    for i, item in enumerate(batch):
        mel_len = item['mel'].size(1)
        mels[i, :, :mel_len] = item['mel']
        stop_targets[i, :mel_len] = item['stop_target']

    # Pad text sequences
    transcript_lengths = [seq.size(0) for seq in transcripts]
    padded_transcripts = pad_sequence(transcripts, batch_first=True, padding_value=0)

    return {
        'transcripts': padded_transcripts,
        'transcript_lengths': torch.tensor(transcript_lengths),
        'mels': mels,
        'embeddings': embeddings,
        'stop_targets': stop_targets,
    }

# Example usage
if __name__ == "__main__":
    # Define character vocabulary
    vocab = {char: idx for idx, char in enumerate("abcdefghijklmnopqrstuvwxyz'.,?! ")}
    vocab['<pad>'] = len(vocab)  # Add padding token

    # Path to metadata
    metadata_path = "data/train/metadata.csv"  # Update with your actual metadata file path

    # Create dataset and dataloader
    dataset = TTSDataLoader(metadata_path, vocab)
    batch_size = 16
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)

    # Iterate through DataLoader
    for batch in dataloader:
        print(f"Padded Transcripts: {batch['transcripts'].shape}")  # (batch_size, max_seq_len)
        print(f"Transcript Lengths: {batch['transcript_lengths']}")  # Lengths of original transcripts
        print(f"Mel shape: {batch['mels'].shape}")  # (batch_size, n_mels, max_mel_len)
        print(f"Embeddings shape: {batch['embeddings'].shape}")  # (batch_size, embedding_dim)
        print(f"Stop targets shape: {batch['stop_targets'].shape}")  # (batch_size, max_mel_len)
        break
