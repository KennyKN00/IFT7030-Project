from datetime import datetime
import time
from functools import partial
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

from synthesizer import audio
from synthesizer.models.tacotron import Tacotron
from synthesizer.synthesizer_dataset import SynthesizerDataset, collate_synthesizer
from synthesizer.utils import ValueWindow, data_parallel_workaround
from synthesizer.utils.plot import plot_spectrogram
from synthesizer.utils.symbols import symbols
from synthesizer.utils.text import sequence_to_text


def np_now(x: torch.Tensor): return x.detach().cpu().numpy()


def time_string():
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def train(run_id: str, syn_dir: Path, models_dir: Path, save_every: int, backup_every: int, force_restart: bool,
          hparams, unfreeze_schedule: dict):
    models_dir.mkdir(exist_ok=True)

    model_dir = models_dir.joinpath(run_id)
    plot_dir = model_dir.joinpath("plots")
    wav_dir = model_dir.joinpath("wavs")
    mel_output_dir = model_dir.joinpath("mel-spectrograms")
    meta_folder = model_dir.joinpath("metas")
    model_dir.mkdir(exist_ok=True)
    plot_dir.mkdir(exist_ok=True)
    wav_dir.mkdir(exist_ok=True)
    mel_output_dir.mkdir(exist_ok=True)
    meta_folder.mkdir(exist_ok=True)

    weights_fpath = model_dir / f"synthesizer.pt"
    metadata_fpath = Path(syn_dir)

    print("Checkpoint path: {}".format(weights_fpath))
    print("Loading training data from: {}".format(metadata_fpath))
    print("Using model: Tacotron")

    time_window = ValueWindow(100)
    loss_window = ValueWindow(100)

    # From WaveRNN/train_tacotron.py
    if torch.cuda.is_available():
        device = torch.device("cuda")

        for session in hparams.tts_schedule:
            _, _, _, batch_size = session
            if batch_size % torch.cuda.device_count() != 0:
                raise ValueError("`batch_size` must be evenly divisible by n_gpus!")
    else:
        device = torch.device("cpu")
    print("Using device:", device)

    # Instantiate Tacotron Model
    print("\nInitializing Tacotron Model...\n")
    model = Tacotron(embed_dims=hparams.tts_embed_dims,
                     num_chars=len(symbols),
                     encoder_dims=hparams.tts_encoder_dims,
                     decoder_dims=hparams.tts_decoder_dims,
                     n_mels=hparams.num_mels,
                     fft_bins=hparams.num_mels,
                     postnet_dims=hparams.tts_postnet_dims,
                     encoder_K=hparams.tts_encoder_K,
                     lstm_dims=hparams.tts_lstm_dims,
                     postnet_K=hparams.tts_postnet_K,
                     num_highways=hparams.tts_num_highways,
                     dropout=hparams.tts_dropout,
                     stop_threshold=hparams.tts_stop_threshold,
                     speaker_embedding_size=hparams.speaker_embedding_size).to(device)

    # Gradual unfreezing logic
    print("\nApplying gradual unfreezing schedule...")
    for name, param in model.named_parameters():
        param.requires_grad = False  # Freeze all layers initially
    print("All layers are initially frozen.")

    # Optimizer with layer-wise learning rates
    optimizer_params = []
    for layer, lr in unfreeze_schedule.items():
        for name, param in model.named_parameters():
            if layer in name:
                param.requires_grad = True
                optimizer_params.append({'params': param, 'lr': lr})
                print(f"Layer {name} is now trainable with learning rate {lr}")

    optimizer = optim.Adam(optimizer_params)

    # Load the weights
    if force_restart or not weights_fpath.exists():
        print("\nStarting the training of Tacotron from scratch\n")
        model.save(weights_fpath)

        # Embeddings metadata
        char_embedding_fpath = meta_folder.joinpath("CharacterEmbeddings.tsv")
        with open(char_embedding_fpath, "w", encoding="utf-8") as f:
            for symbol in symbols:
                if symbol == " ":
                    symbol = "\\s"  # For visual purposes, swap space with \s
                f.write("{}\n".format(symbol))
    else:
        print("\nLoading weights at %s" % weights_fpath)
        model.load(weights_fpath) # Reset optimizer 
        model.register_buffer("step", torch.tensor(0)) 
        print("Tacotron weights loaded from step %d" % model.step)

    # Dataset and DataLoader initialization
    dataset = SynthesizerDataset(metadata_fpath, hparams)
    print('load dataset')
    # Training loop
    for i, session in enumerate(hparams.tts_schedule):
        
        current_step = model.get_step()

        r, lr, max_step, batch_size = session
        training_steps = max_step - current_step

        if current_step >= max_step:
            if i == len(hparams.tts_schedule) - 1:
                model.save(weights_fpath, optimizer)
                break
            else:
                continue
        model.r = r

        simple_table([(f"Steps with r={r}", str(training_steps // 1000) + "k Steps"),
                      ("Batch Size", batch_size),
                      ("Learning Rate", lr),
                      ("Outputs/Step (r)", model.r)])

        for p in optimizer.param_groups:
            p["lr"] = lr

        collate_fn = partial(collate_synthesizer, r=r, hparams=hparams)
        data_loader = DataLoader(dataset, batch_size, shuffle=True, num_workers=2, collate_fn=collate_fn)

        total_iters = len(dataset)
        steps_per_epoch = np.ceil(total_iters / batch_size).astype(np.int32)
        epochs = np.ceil(training_steps / steps_per_epoch).astype(np.int32)

        for epoch in range(1, epochs + 1):
            # Dynamically unfreeze layers based on schedule
            for layer, epoch_to_unfreeze in unfreeze_schedule.items():
                if epoch >= epoch_to_unfreeze:
                    for name, param in model.named_parameters():
                        if layer in name and not param.requires_grad:
                            param.requires_grad = True
                            print(f"Unfroze layer {name} at epoch {epoch}")


            for i, (texts, mels, embeds, idx) in enumerate(data_loader, 1):
                start_time = time.time()

                # Generate stop tokens for training
                stop = torch.ones(mels.shape[0], mels.shape[2])
                for j, k in enumerate(idx):
                    mel_path = dataset.metadata.iloc[k, 4]  
                    stop_frame_length = torch.load(mel_path).shape[-1]
                    stop[j, :int(stop_frame_length) - 1] = 0

                texts = torch.tensor(texts).to(device)
                mels = mels.to(device)
                embeds = embeds.to(device)
                stop = stop.to(device)

                # Forward pass
                # Parallelize model onto GPUS using workaround due to python bug
                if device.type == "cuda" and torch.cuda.device_count() > 1:
                    m1_hat, m2_hat, attention, stop_pred = data_parallel_workaround(model, texts, mels, embeds)
                else:
                    m1_hat, m2_hat, attention, stop_pred = model(texts, mels, embeds)

                # Backward pass
                m1_loss = F.mse_loss(m1_hat, mels) + F.l1_loss(m1_hat, mels)
                m2_loss = F.mse_loss(m2_hat, mels)
                stop_loss = F.binary_cross_entropy(stop_pred, stop)

                loss = m1_loss + m2_loss + stop_loss

                optimizer.zero_grad()
                loss.backward()

                if hparams.tts_clip_grad_norm is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), hparams.tts_clip_grad_norm)
                    if np.isnan(grad_norm.cpu()):
                        print("grad_norm was NaN!")

                optimizer.step()

                time_window.append(time.time() - start_time)
                loss_window.append(loss.item())

                step = model.get_step()
                k = step // 1000

                msg = f"| Epoch: {epoch}/{epochs} ({i}/{steps_per_epoch}) | Loss: {loss_window.average:#.4} | " \
                    f"{1./time_window.average:#.2} steps/s | Step: {k}k | "
                stream(msg)

                # Backup or save model as appropriate
                if backup_every != 0 and step % backup_every == 0 :
                    backup_fpath = weights_fpath.parent / f"synthesizer_{k:06d}.pt"
                    model.save(backup_fpath, optimizer)

                if save_every != 0 and step % save_every == 0 :
                    # Must save latest optimizer state to ensure that resuming training
                    # doesn't produce artifacts
                    model.save(weights_fpath, optimizer)

                # Evaluate model to generate samples
                epoch_eval = hparams.tts_eval_interval == -1 and i == steps_per_epoch  # If epoch is done
                step_eval = hparams.tts_eval_interval > 0 and step % hparams.tts_eval_interval == 0  # Every N steps
                if epoch_eval or step_eval:
                    for sample_idx in range(hparams.tts_eval_num_samples):
                        # At most, generate samples equal to number in the batch
                        if sample_idx + 1 <= len(texts):
                            # Remove padding from mels using frame length in metadata
                            mel_path = dataset.metadata.iloc[idx[sample_idx], 4]
                            mel_length = torch.load(mel_path).shape[-1]
                            mel_prediction = np_now(m2_hat[sample_idx]).T[:mel_length]
                            target_spectrogram = np_now(mels[sample_idx]).T[:mel_length]
                            attention_len = mel_length // model.r

                            eval_model(attention=np_now(attention[sample_idx][:, :attention_len]),
                                    mel_prediction=mel_prediction,
                                    target_spectrogram=target_spectrogram,
                                    input_seq=np_now(texts[sample_idx]),
                                    step=step,
                                    plot_dir=plot_dir,
                                    mel_output_dir=mel_output_dir,
                                    wav_dir=wav_dir,
                                    sample_num=sample_idx + 1,
                                    loss=loss,
                                    hparams=hparams)

                # Break out of loop to update training schedule
                if step >= max_step:
                    break

            # Add line break after every epoch
            print("")


def eval_model(attention, mel_prediction, target_spectrogram, input_seq, step,
               plot_dir, mel_output_dir, wav_dir, sample_num, loss, hparams):
    # Save some results for evaluation
    attention_path = str(plot_dir.joinpath("attention_step_{}_sample_{}".format(step, sample_num)))
    save_attention(attention, attention_path)

    # save predicted mel spectrogram to disk (debug)
    mel_output_fpath = mel_output_dir.joinpath("mel-prediction-step-{}_sample_{}.npy".format(step, sample_num))
    np.save(str(mel_output_fpath), mel_prediction, allow_pickle=False)

    # save griffin lim inverted wav for debug (mel -> wav)
    wav = audio.inv_mel_spectrogram(mel_prediction.T, hparams)
    wav_fpath = wav_dir.joinpath("step-{}-wave-from-mel_sample_{}.wav".format(step, sample_num))
    audio.save_wav(wav, str(wav_fpath), sr=hparams.sample_rate)

    # save real and predicted mel-spectrogram plot to disk (control purposes)
    spec_fpath = plot_dir.joinpath("step-{}-mel-spectrogram_sample_{}.png".format(step, sample_num))
    title_str = "{}, {}, step={}, loss={:.5f}".format("Tacotron", time_string(), step, loss)
    plot_spectrogram(mel_prediction, str(spec_fpath), title=title_str,
                     target_spectrogram=target_spectrogram,
                     max_len=target_spectrogram.size // hparams.num_mels)
    print("Input at step {}: {}".format(step, sequence_to_text(input_seq)))

def stream(message) :
    try:
        sys.stdout.write("\r{%s}" % message)
    except:
        #Remove non-ASCII characters from message
        message = ''.join(i for i in message if ord(i)<128)
        sys.stdout.write("\r{%s}" % message)

def simple_table(item_tuples) :

    border_pattern = '+---------------------------------------'
    whitespace = '                                            '

    headings, cells, = [], []

    for item in item_tuples :

        heading, cell = str(item[0]), str(item[1])

        pad_head = True if len(heading) < len(cell) else False

        pad = abs(len(heading) - len(cell))
        pad = whitespace[:pad]

        pad_left = pad[:len(pad)//2]
        pad_right = pad[len(pad)//2:]

        if pad_head :
            heading = pad_left + heading + pad_right
        else :
            cell = pad_left + cell + pad_right

        headings += [heading]
        cells += [cell]

    border, head, body = '', '', ''

    for i in range(len(item_tuples)) :

        temp_head = f'| {headings[i]} '
        temp_body = f'| {cells[i]} '

        border += border_pattern[:len(temp_head)]
        head += temp_head
        body += temp_body

        if i == len(item_tuples) - 1 :
            head += '|'
            body += '|'
            border += '+'

    print(border)
    print(head)
    print(border)
    print(body)
    print(border)
    print(' ')

def save_attention(attn, path):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12, 6))
    plt.imshow(attn.T, interpolation='nearest', aspect='auto')
    fig.savefig(f'{path}.png', bbox_inches='tight')
    plt.close(fig)