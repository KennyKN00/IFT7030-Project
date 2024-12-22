# Zero-Shot Voice Cloning with Fine-Tuned Models

This repository contains the code and methodology for fine-tuning pretrained models to achieve zero-shot voice cloning. The project uses components from the Real-Time Voice Cloning project, including a speaker encoder, synthesizer, and HiFi-GAN vocoder.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Setup and Installation](#setup-and-installation)
3. [Data Preparation](#data-preparation)
4. [Training](#training)
5. [Inference and Evaluation](#inference-and-evaluation)


## Project Overview

Zero-shot voice cloning allows for synthesizing a target speaker's voice using minimal or no additional training data. This project focuses on fine-tuning pretrained models to improve performance using dynamic layer freezing and multi-speaker datasets.

Key features:
- Fine-tuning the synthesizer with dynamic layer freezing.
- Comparing single-speaker and multi-speaker setups.
- Generating high-quality audio using HiFi-GAN vocoder.
- Evaluating performance with Word Error Rate (WER), Mel Cepstral Distortion (MCD), and Mean Opinion Score (MOS).


## Setup and Installation

### Prerequisites
- Python 3.8 or above
- PyTorch 1.10 or above
- NVIDIA GPU with CUDA 11 support

### Installation
1. Clone this repository :
    ```bash
    git clone https://github.com/your-repo/zero-shot-voice-cloning.git
    cd zero-shot-voice-cloning
    ```

2. Create a virtual environment and install dependencies :
    ```bash
    python -m venv venv
    source venv/bin/activate  # For Linux/Mac
    venv\Scripts\activate     # For Windows
    pip install -r requirements.txt
    ```

3. Download the pretrained models and place them into the `models/` directory :
- Speaker Encoder : [The encoder.pt file is here](https://drive.google.com/drive/folders/1HnPKj61FoDJwMLTHCgRnfccz3OyhQikg?usp=sharing)
- HiFi-GAN vocoder : [The vocoder.pt file is here](https://drive.google.com/drive/folders/1HnPKj61FoDJwMLTHCgRnfccz3OyhQikg?usp=sharing)
- Fine-tuned checkpoint for first method : [The checkpoint_epoch_2050.pt is here](https://drive.google.com/drive/folders/1HnPKj61FoDJwMLTHCgRnfccz3OyhQikg?usp=sharing)
- Fine-tuned checkpoint for second method : [The synthesizer_000030.pt is here](https://drive.google.com/drive/folders/1HnPKj61FoDJwMLTHCgRnfccz3OyhQikg?usp=sharing)

## Data preparation

The training and validation dataset are already preprocess and save under the `data` directory as `train` and `val`. Each one of them has the following structure :
```
dataset/
├── mel_spectrograms
|   ├── speaker_id/
|   |   ├── mel_id.pt
|   |   ├── mel_id.pt
|   |   ├── mel_id.pt
|   ├── speaker_id/
|   |   ...
├── audios
|   ├── speaker_id/
|   |   ├── audio_id.flac
|   |   ├── audio_id.flac
|   |   ├── audio_id.flac
|   ├── speaker_id/
|   |   |   ...
├── speaker_embeddings
|   ├── speaker_id/
|   |   ├── embedding_id.pt
|   |   ├── embedding_id.pt
|   |   ├── embedding_id.pt
|   ├── speaker_id/
|   |   |   ...
└── metadata.csv
```

## Training 

To train the model, un-comment the `Fine-tuning 1` part for the first implementation of fine-tuning. Else, un-comment `Fine-tuning 2` part for the second implementation of a fine-tuning.

- Checkpoints will be saved in the `models/` directory.

## Inference and Evaluation

### Generate Audio
To generate the audio, un-comment the `Get mel before fine-tuning` and you will get a mel-spectrogram and the syntehsized audio.

Similarly, un-comment the `Get mel after fine-tuning` to output the synthesized audio and mel-spectrogram.

### Evaluate Performance
Once a synthesized audio is output, run the script in `evaluation/evaluation.py` to get both MCD and WER score. You can proceed with this step after geneerating audio before and after fine-tuning.