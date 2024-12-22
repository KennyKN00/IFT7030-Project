import torch
from dataloader import TTSDataLoader, collate_fn
from synthesizer.models.tacotron import Tacotron
from synthesizer.utils.text import text_to_sequence
from pathlib import Path
import pandas as pd
from torch.utils.data import DataLoader
from synthesizer.utils.symbols import symbols
from torch.nn.utils.rnn import pad_sequence
from matplotlib import pyplot as plt
import torch.nn.functional as F


# Define paths
PRETRAINED_SYNTHESIZER_PATH = "models/synthesizer.pt"
PROCESSED_DATA_PATH = "data/preprocessed_data_test"  
TRAIN_METADATA_PATH = "data/train/metadata.csv"
VAL_METADATA_PATH = "data/val/metadata.csv"
SAVE_PATH = "models/fine_tuned_synthesizer.pt"

# Hyperparameters
EPOCHS = 5000
LEARNING_RATE = 1e-5
BATCH_SIZE = 16

def get_loss_curves(epochs, train_losses, val_losses):
    x = range(1, epochs+1)

    plt.figure()    
    plt.plot(x, train_losses, label='Training Loss')
    plt.plot(x, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curves')
    plt.legend()
    plt.savefig('images/Loss_after.png') 
    plt.show()


def fine_tune(synthesizer, train_loader, val_loader, epochs=EPOCHS, lr=LEARNING_RATE, freeze_layers=True, patience=5, checkpoint_path=None):
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
    val_losses = []

    best_val_loss = float("inf")
    patience_counter = 0
    number_epochs = 0
    start_epoch = 1

    # ---- Load Checkpoint if Provided ---- #
    if checkpoint_path:
        print(f"INFO: Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        synthesizer.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint["epoch"] + 1  # Resume from next epoch
        print(f"INFO: Resuming training from epoch {start_epoch}")
    
    print("INFO : Starting fine-tuning")
    for epoch in range(start_epoch, epochs + 1):
        train_loss = 0
        num_batches = 0

        # Training Loop
        synthesizer.train()  
        for batch in train_loader:
            optimizer.zero_grad()

            mels = batch["mels"].to(device)
            texts = batch["transcripts"]
            embeds = batch["embeddings"].to(device)
            
            # Forward pass
            mel_outputs, mel_outputs_postnet, alignments, stop_outputs = synthesizer(x=texts, m=mels, speaker_embedding=embeds)

            min_len = min(mel_outputs.size(2), mels.size(2))
            mel_outputs = mel_outputs[:, :, :min_len]
            mel_outputs_postnet = mel_outputs_postnet[:, :, :min_len]
            mels = mels[:, :, :min_len]

            # Compute training losses
            mel_loss = F.mse_loss(mel_outputs, mels) + F.l1_loss(mel_outputs, mels)
            mel_postnet_loss = F.mse_loss(mel_outputs_postnet, mels)
            loss = mel_loss + mel_postnet_loss 

            # Backward pass and optimization
            loss.backward()
            torch.nn.utils.clip_grad_norm_(synthesizer.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        train_loss /= num_batches
        train_losses.append(train_loss)
        print(f"Epoch {epoch}, Training Loss: {train_loss:.4f}")

        # Validation Loop
        synthesizer.eval()  
        val_loss = 0
        num_val_batches = 0
        with torch.no_grad():  
            for batch in val_loader:
                optimizer.zero_grad()

                mels = batch["mels"].to(device)
                texts = batch["transcripts"]
                embeds = batch["embeddings"].to(device)
                
                # Forward pass
                mel_outputs, mel_outputs_postnet, alignments, stop_outputs = synthesizer(x=texts, m=mels, speaker_embedding=embeds)

                min_len = min(mel_outputs.size(2), mels.size(2))
                mel_outputs = mel_outputs[:, :, :min_len]
                mel_outputs_postnet = mel_outputs_postnet[:, :, :min_len]
                mels = mels[:, :, :min_len]

                # Compute validation losses
                mel_loss = F.mse_loss(mel_outputs, mels) + F.l1_loss(mel_outputs, mels)
                mel_postnet_loss = F.mse_loss(mel_outputs_postnet, mels)
                loss = mel_loss + mel_postnet_loss 

                val_loss += loss.item()
                num_val_batches += 1

        val_loss /= num_val_batches
        val_losses.append(val_loss)
        print(f"Epoch {epoch}, Validation Loss: {val_loss:.4f}")

        number_epochs += 1
        # Save Model Checkpoint
        if epoch % 10 == 0 : 
            torch.save({
                "model_state": synthesizer.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch,
            }, f"models/checkpoint_epoch_{epoch}.pt")

    get_loss_curves(number_epochs, train_losses, val_losses)
    print(f"Fine-tuned model saved to {SAVE_PATH}")
    
def run():
    # Define character vocabulary
    vocab = {char: idx for idx, char in enumerate("abcdefghijklmnopqrstuvwxyz'.,?! ")}
    vocab['<pad>'] = len(vocab)
        
    traininig_dataset = TTSDataLoader(TRAIN_METADATA_PATH, vocab)
    training_loader = DataLoader(traininig_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, collate_fn=collate_fn)

    val_dataset = TTSDataLoader(VAL_METADATA_PATH, vocab)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, collate_fn=collate_fn)

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
    synthesizer.reset_step()
    synthesizer.load(PRETRAINED_SYNTHESIZER_PATH)

    print("Model initialized")
    
    fine_tune(synthesizer, training_loader, val_loader)