from matplotlib import pyplot as plt
import torchaudio
import torch
import torch.nn.functional as F
from speechbrain.inference.TTS import MSTacotron2
from speechbrain.inference.vocoders import HIFIGAN
from dataset_loader import dataloader

# Intialize TTS - Needs to be run in admin
ms_tacotron2 = MSTacotron2.from_hparams(source="speechbrain/tts-mstacotron2-libritts", savedir="pretrained_models/tts-mstacotron2-libritts")
hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-libritts-22050Hz", savedir="pretrained_models/tts-hifigan-libritts-22050Hz")

for name, param in ms_tacotron2.mods["model"].named_parameters():
    if "encoder" in name or "speaker_embedding" in name:
        param.requires_grad = False  # Freeze encoder and speaker embedding layers
    elif "decoder" in name or "attention" in name:
        param.requires_grad = True  # Keep attention and decoder layers trainable

def get_loss_curves(epochs, train_losses):
    x = range(1, epochs+1)
    y = train_losses
    
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curves')
    plt.legend()
    plt.show()
    
    
def fine_tune_model(model, dataloader, num_epochs=5, learning_rate=1e-4):
    
    # Use only the unfrozen parameters for optimization
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

    model.train()  # Set the model to training mode
    train_losses = []

    for epoch in range(num_epochs):
        train_loss = 0.0
        num_batches = 0
        
        for batch in dataloader:
            # Unpack the batch
            mel_targets = batch["mel"]  # Target mel-spectrograms
            transcripts = batch["transcript"]  # Transcripts
            speaker_ids = batch["speaker_id"]  # Speaker IDs

            # Compute speaker embeddings
            spk_embs = []
            for sid in speaker_ids:
                spk_emb = ms_tacotron2.spk_emb_encoder.encode_mel_spectrogram_batch(batch["mel"]).squeeze(0)
                spk_embs.append(spk_emb)
            spk_embs = torch.stack(spk_embs)
            spk_embs = spk_embs.view(spk_embs.size(0), -1, spk_embs.size(-1)).squeeze(1)
            spk_embs = spk_embs.mean(dim=1)  # Collapse the second dimension by averaging

            # Convert transcripts to phoneme sequences
            phoneme_seqs = []
            for transcript in transcripts:
                phoneme_seq = ms_tacotron2.g2p([transcript])[0]
                phoneme_seq = "{" + " ".join(phoneme_seq) + "}"
                phoneme_seqs.append(phoneme_seq)
                
            # Generate mel-spectrogram predictions
            mel_outputs, mel_lengths, alignments = ms_tacotron2._MSTacotron2__encode_batch(
                phoneme_seqs, spk_embs
            )
            
            if mel_outputs.size(2) < mel_targets.size(2):
                padding = mel_targets.size(2) - mel_outputs.size(2)
                mel_outputs = F.pad(mel_outputs, (0, padding), mode='constant', value=0.0)
            else:
                mel_outputs = mel_outputs[:, :, :mel_targets.size(2)]

            # Compute loss 
            loss = F.mse_loss(mel_outputs, mel_targets)
            loss.requires_grad = True
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches +1
            print(f"Epoch {epoch + 1}, Loss: {loss.item()}")
            
        train_loss /= num_batches
        train_losses.append(train_loss)
        
        if epoch in [2, 4, 9]:
            save_path = f"models/fine_tuned_ms_tacotron2_epoch_{epoch}.pth"
            torch.save(ms_tacotron2.state_dict(), save_path)
            print(f"Model saved at {save_path}")
    
    get_loss_curves(num_epochs, train_losses)
        

fine_tune_model(ms_tacotron2, dataloader, num_epochs=10, learning_rate=1e-4)