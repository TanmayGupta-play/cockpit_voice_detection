# inference_trigger.py
import sys
import os

# Add project root (adjust the path if necessary)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np
from models.trigger_word import TriggerCRNN
from utils.preprocessing import extract_log_mel

# -------------------------------
# Configuration
# -------------------------------
MODEL_PATH = "checkpoints/trigger_word_model.pth"
SAMPLE_RATE = 16000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# Load Trained Model
# -------------------------------
model = TriggerCRNN().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# -------------------------------
# Prediction Function
# -------------------------------
def predict_trigger(audio_path, threshold=0.4):
    """
    Predict trigger word presence in a given audio.

    Args:
        audio_path (str): Path to .wav file
        threshold (float): Probability threshold (default = 0.5)

    Returns:
        int: 1 if trigger detected, else 0
        float: probability score
    """
    log_mel = extract_log_mel(audio_path)  # [64, T]
    log_mel_tensor = torch.tensor(log_mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)  # [1, 1, 64, T]

    with torch.no_grad():
        output = model(log_mel_tensor)  # [1, 1]
        prob = output.item()
        prediction = int(prob >= threshold)

    return prediction, prob

# -------------------------------
# Example Run
# -------------------------------
if __name__ == "__main__":
    test_audio = "sample_inputs/test_trigger.wav"  # Replace with your file
    pred, prob = predict_trigger(test_audio)
    print(f"🔍 Trigger Detected: {bool(pred)} | Confidence: {prob:.4f}")