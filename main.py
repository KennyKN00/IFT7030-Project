import torch
from pathlib import Path
from encoder import audio
from encoder import inference 
from preprocessing.recorder import Recorder

def main():
    # Path to the pre-trained encoder weights
    encoder_model_path = Path("models/encoder.pt")

    # Load the model using load_model, which sets _model globally
    _model = inference.load_model(encoder_model_path, device="cuda" if torch.cuda.is_available() else "cpu")

    if _model is None:
            raise ValueError("The encoder model was not loaded correctly.")
        
    # Pass _model and audio_module to Recorder
    recorder = Recorder(encoder=_model, inference=inference, audio=audio)

    # Run the recording and processing workflow
    recorder.run()
    
    # data = torch.load("mel_spectrograms/001.pt")

    # # Check keys and structure
    # print("Keys in .pt file:", data.keys())
    # for key, value in data.items():
    #     print(f"{key}: {type(value)}, shape: {value.shape if hasattr(value, 'shape') else 'N/A'}")
    #     print(value)
    
    

if __name__ == '__main__':
    main()