# Zero-Shot Voice Cloning with Fine-Tuned Models

This repository contains the code and methodology for fine-tuning pretrained models to achieve zero-shot voice cloning. The project uses components from the Real-Time Voice Cloning project, including a speaker encoder, synthesizer, and HiFi-GAN vocoder.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Setup and Installation](#setup-and-installation)
3. [Data Preparation](#data-preparation)
4. [Training](#training)
5. [Inference and Evaluation](#inference-and-evaluation)
6. [Results](#results)
7. [Acknowledgments](#acknowledgments)


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

To train the model, run the training script with the following command :
``` bash
python train.py --data-dir ./preprocessed_data \
                --output-dir ./checkpoints \
                --batch-size 32 \
                --epochs 50 \
                --learning-rate 1e-4 \
                --freeze-encoder True \
                --dynamic-freezing True
```
- Flags :
    - --freeze-encoder: Freezes the encoder layers.
    - --dynamic-freezing: Gradually unfreezes layers during training.

- Checkpoints will be saved in the `models/` directory.

## Inference and Evaluation

### Generate Audio
Use the `inference.py` script to synthesize audio :
``` bash
python inference.py --input-text "Hello, this is a zero-shot voice cloning test." \
                    --speaker-embedding ./preprocessed_data/speaker1/embedding.pt \
                    --output-audio ./output_audio.wav

```

### Evaluate Performance
Run the evaluation script to calculate WER and MCD.
``` bash
python evaluate.py --generated-audio ./output_audio.wav \
                   --ground-truth-audio ./datasets/speaker1/audio1.wav

```