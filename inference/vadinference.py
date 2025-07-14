# inference/vad_infer.py
import sys
import os

# Add project root (adjust the path if necessary)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torchaudio
import numpy as np
from models.vad_crnn import VADCRNN
from utils.preprocessing import extract_log_mel

# Config
VAD_MODEL_PATH = "checkpoints/vad_crnn.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
vad_model = VADCRNN().to(DEVICE)
vad_model.load_state_dict(torch.load(VAD_MODEL_PATH, map_location=DEVICE))
vad_model.eval()

def is_speech(audio_path, threshold=0.5):
    """Predicts if speech is present in the audio."""
    # ✅ Load directly from file path using librosa inside extract_log_mel
    mel_spec = extract_log_mel(audio_path, sr=16000)  # [64, T]
    mel_spec_tensor = torch.tensor(mel_spec, dtype=torch.float32).unsqueeze(0).to(DEVICE)  # [1, 1, 64, T]

    with torch.no_grad():
        output = vad_model(mel_spec_tensor)
        prediction = output.item() > threshold

    return prediction, output.item()

# Example usage
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python vad_infer.py <audio_file.wav>")
    else:
        audio_file = sys.argv[1]
        result, prob = is_speech(audio_file)
        print(f"🔊 Speech Detected: {result} (Confidence: {prob:.3f})")