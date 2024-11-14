import glob
import os
import torch
import torch.nn as nn
from datasetLoader import Tacotron2FineTuneDataset
from torch.utils.data import DataLoader
from multi_speaker_model import MultiSpeakerTacotron2

# Load the pretrained Tacotron 2 model from torch.hub
tacotron2 = torch.hub.load('nvidia/DeepLearningExamples:torchhub', 'nvidia_tacotron2')

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
multi_speaker_tacotron2 = MultiSpeakerTacotron2(tacotron2).to(device)

# Define optimizer and loss function
optimizer = torch.optim.Adam(tacotron2.parameters(), lr=1e-4)
criterion = nn.MSELoss()  # Mean Squared Error for mel spectrogram prediction

# List of .pt file paths (adjust to your files)
pt_file_paths = glob.glob(os.path.join("mel_spectrograms", "*.pt"))
pt_file_paths = [path.replace("\\", "/") for path in pt_file_paths] # Windows backslash into slash

# Create the dataset and dataloader
dataset = Tacotron2FineTuneDataset(pt_file_paths)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

num_epochs = 2  # Number of epochs for fine-tuning

for epoch in range(num_epochs):
    epoch_loss = 0.0
    for transcription_ids, mel_spectrogram, speaker_embedding in dataloader:
        # Move inputs to device
        transcription_ids = transcription_ids.to(device)
        mel_spectrogram = mel_spectrogram.to(device)
        speaker_embedding = speaker_embedding.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass through Tacotron 2
        predicted_mel, predicted_mel_postnet, gate_outputs, alignments = multi_speaker_tacotron2(transcription_ids, speaker_embedding)

        # Calculate loss
        loss = criterion(predicted_mel, mel_spectrogram) + criterion(predicted_mel_postnet, mel_spectrogram)
        epoch_loss += loss.item()

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

    # Print epoch loss
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss / len(dataloader)}")
    
# torch.save(tacotron2.state_dict(), "fine_tuned_tacotron2.pth")
print('Training complete')