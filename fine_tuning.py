import torch
from synthesizer.models.tacotron import Tacotron
from synthesizer.utils.text import text_to_sequence
from pathlib import Path
import pandas as pd
from torch.utils.data import DataLoader
from synthesizer.utils.symbols import symbols
from torch.nn.utils.rnn import pad_sequence
from matplotlib import pyplot as plt


# Define paths
PRETRAINED_SYNTHESIZER_PATH = "pretrained_models/synthesizer.pt"
PROCESSED_DATA_PATH = "data/preprocessed_data_test"  
METADATA_PATH = "data/metadata.csv"
SAVE_PATH = "models/fine_tuned_synthesizer.pt"

# Hyperparameters
EPOCHS = 10
LEARNING_RATE = 1e-4
BATCH_SIZE = 16

def get_loss_curves(epochs, train_losses):
    x = range(1, epochs+1)
    y = train_losses
    
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curves')
    plt.legend()
    plt.savefig('my_plot.png') 
    plt.show()


def load_batch(batch):
    # Extract mel-spectrograms, transcripts, and speaker IDs
    mel_targets = [item["mel"].transpose(0,1) for item in batch] 
    transcripts = [item["transcript"] for item in batch]
    speaker_embeddings = [item["speaker_embeddings"] for item in batch]
    speaker_embeddings = torch.stack(speaker_embeddings, dim=0)

    # Pad mel-spectrograms along the time dimension
    mel_targets_padded = pad_sequence(
        mel_targets,  # Transpose each to [time, n_mels]
        batch_first=True,
        padding_value=0.0
    )
    mel_targets_padded = mel_targets_padded.transpose(1, 2)  # Back to [batch, frequency, max_time]

    return {
        "mel": mel_targets_padded,
        "transcript": transcripts,
        "speaker_embeddings": speaker_embeddings,
    }

class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, metadata):
        self.metadata = metadata

    def __len__(self):
        return len(self.metadata)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        mel = torch.tensor(torch.load(row["mel_path"]))
        transcript = row["transcript"]
        speaker_embeddings = torch.tensor(torch.load(row["speaker_embedding_path"]), dtype=torch.float32)
        return {"mel": mel, "transcript": transcript, "speaker_embeddings": speaker_embeddings}


def fine_tune(synthesizer, dataloader, epochs=EPOCHS, lr=LEARNING_RATE, freeze_layers=True):
    # Freeze encoder layers to retain zero-shot performance
    if freeze_layers:
        for name, param in synthesizer.named_parameters():
            if "encoder" in name:  # Freeze encoder-related layers
                param.requires_grad = False

    # Send to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    synthesizer.to(device)
    synthesizer.train() 

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, synthesizer.parameters()), lr=lr)
    criterion = torch.nn.MSELoss()
    train_losses = []
    
    print("INFO : Starting fine-tuning")
    for epoch in range(epochs):
        train_loss = 0
        num_batches = 0
        
        for batch in dataloader:
            optimizer.zero_grad()

            mel_specs = batch["mel"].to(device)
            texts = batch["transcript"]
            speaker_embeddings = batch["speaker_embeddings"].to(device)
            
            text_sequences = [torch.tensor(text_to_sequence(text, ["english_cleaners"]), dtype=torch.long) for text in texts]
            text_sequences = pad_sequence(text_sequences, batch_first=True).to(device)  
            
            # Forward pass
            mel_outputs, mel_outputs_postnet, alignments, stop_outputs = synthesizer(
                x=text_sequences,
                m=mel_specs,
                speaker_embedding=speaker_embeddings
            )
            
            min_len = min(mel_outputs.size(2), mel_specs.size(2))
            mel_outputs = mel_outputs[:, :, :min_len]
            mel_outputs_postnet = mel_outputs_postnet[:, :, :min_len]
            mel_specs = mel_specs[:, :, :min_len]

            # Compute loss 
            loss = criterion(mel_outputs, mel_specs) + criterion(mel_outputs_postnet, mel_specs)
            train_loss += loss.item()
            num_batches += 1

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {train_loss / len(dataloader)}")
        train_loss /= num_batches
        train_losses.append(train_loss)
        
        if epoch in [2, 4, 9]:
            checkpoint_path = f"models/fine_tuned_synthesizer_epoch_{epoch}.pth"
            torch.save(synthesizer.state_dict(), checkpoint_path)
            print(f"Model saved at {checkpoint_path}")

    # Save fine-tuned model
    torch.save({
        "model_state_dict": synthesizer.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, SAVE_PATH)

    print(f"Fine-tuned model saved to {SAVE_PATH}")
    
def run():
    metadata = pd.read_csv(METADATA_PATH)
        
    dataset = EpisodicDataset(metadata)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=load_batch)

    params = {
        "embed_dims": 512,
        "num_chars": len(symbols),               
        "encoder_dims": 256,
        "decoder_dims": 128,
        "n_mels": 80,                   
        "fft_bins": 80,               
        "postnet_dims": 512,
        "encoder_K": 5,                
        "lstm_dims": 1024,
        "postnet_K": 5,               
        "num_highways": 4,
        "dropout": 0.5,                 
        "stop_threshold": -3.4,         
        "speaker_embedding_size": 256   
    }

    synthesizer = Tacotron(**params)
    synthesizer.load(PRETRAINED_SYNTHESIZER_PATH)
    print("Model initialized")
    
    fine_tune(synthesizer, dataloader)