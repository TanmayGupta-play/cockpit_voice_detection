import sys
import os

# Add project root (adjust the path if necessary)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np
import librosa
from models.command_asr import ASRModel
from utils.preprocessing import extract_log_mel

# ---------------------------
# Configs
# ---------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "checkpoints/command_asr.pth"
SAMPLE_RATE = 16000
N_MELS = 64

# ---------------------------
# Load Model & Mappings
# ---------------------------
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
char2idx = checkpoint["char2idx"]
idx2char = checkpoint["idx2char"]

model = ASRModel(vocab_size=len(char2idx)).to(DEVICE)
model.load_state_dict(checkpoint["model_state"])
model.eval()

# ---------------------------
# CTC Decoding (Greedy)
# ---------------------------
def greedy_decode(output):
    """
    Converts CTC output logits into text using greedy decoding.
    """
    pred_ids = torch.argmax(output, dim=-1)  # [T]
    prev = -1
    decoded = []
    for p in pred_ids:
        if p.item() != prev and p.item() != 0:  # 0 is <blank>
            decoded.append(idx2char[p.item()])
        prev = p.item()
    return ''.join(decoded)

# ---------------------------
# Inference Function
# ---------------------------
def transcribe_audio(audio_path):
    log_mel = extract_log_mel(audio_path, sr=SAMPLE_RATE, n_mels=N_MELS)  # [64, T]
    log_mel_tensor = torch.tensor(log_mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)  # [1, 1, 64, T]

    with torch.no_grad():
        logits = model(log_mel_tensor)  # [B, T', vocab_size]
        logits = logits.squeeze(0)  # [T', vocab_size]
        transcript = greedy_decode(logits.cpu())
    return transcript

# ---------------------------
# Example Usage
# ---------------------------
if __name__ == "__main__":
    test_file = "sample_inputs/test_command.wav"
    prediction = transcribe_audio(test_file)
    print(f"📝 Predicted Transcript: {prediction}")