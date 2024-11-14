import torch
import torch.nn as nn

class MultiSpeakerTacotron2(nn.Module):
    def __init__(self, tacotron2, speaker_embedding_dim=256):
        super(MultiSpeakerTacotron2, self).__init__()
        self.tacotron2 = tacotron2
        self.speaker_embedding_dim = speaker_embedding_dim

    def forward(self, text_ids, speaker_embedding):
        # Pass text_ids through the original embedding
        text_embedding = self.tacotron2.embedding(text_ids)  # Shape: [batch_size, seq_len, embedding_dim]

        text_embedding = text_embedding.squeeze(1)
        
        # Permute text_embedding to [batch_size, embedding_dim, seq_len] as expected by conv1d
        text_embedding = text_embedding.permute(0, 2, 1)

        # Calculate input_lengths as a 1D CPU int64 tensor
        input_lengths = (text_ids != 0).sum(dim=1).to(torch.int64).view(-1).cpu()

        # Pass the text_embedding and input_lengths to the encoder
        encoder_outputs = self.tacotron2.encoder(text_embedding, input_lengths)

        # Expand speaker embedding to match sequence length and concatenate with encoder outputs
        speaker_embedding_expanded = speaker_embedding.unsqueeze(1).expand(-1, encoder_outputs.size(1), -1)

        # Concatenate speaker embedding with encoder outputs
        combined_inputs = torch.cat((encoder_outputs, speaker_embedding_expanded), dim=-1)

        # Pass the combined input to the Tacotron 2 decoder
        mel_outputs, mel_outputs_postnet, gate_outputs, alignments = self.forward_decoder(combined_inputs)
        
        return mel_outputs, mel_outputs_postnet, gate_outputs, alignments

    def forward_decoder(self, combined_inputs):
        # Initialize hidden states for attention and decoder LSTM cells
        attention_hidden = torch.zeros((combined_inputs.size(0), self.tacotron2.decoder.attention_rnn.hidden_size)).to(combined_inputs.device)
        decoder_hidden = torch.zeros((combined_inputs.size(0), self.tacotron2.decoder.decoder_rnn.hidden_size)).to(combined_inputs.device)

        # Example decoder operation loop for each time step
        mel_outputs = []
        for i in range(combined_inputs.size(1)):
            # At each time step, we use a combined input with speaker embedding
            attention_input = combined_inputs[:, i, :]

            # Run through the attention LSTM cell and then the decoder
            attention_hidden, _ = self.tacotron2.decoder.attention_rnn(attention_input, (attention_hidden, attention_hidden))
            decoder_hidden, _ = self.tacotron2.decoder.decoder_rnn(attention_hidden, (decoder_hidden, decoder_hidden))

            mel_output = self.tacotron2.decoder.linear_projection(decoder_hidden)
            mel_outputs.append(mel_output)

        # Stack time steps into final mel output
        mel_outputs = torch.stack(mel_outputs, dim=1)
        mel_outputs_postnet = self.tacotron2.postnet(mel_outputs)
        return mel_outputs, mel_outputs_postnet, None, None  # Simplified without gates and alignments