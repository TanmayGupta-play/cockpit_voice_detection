import os
import librosa
import numpy as np
import json
import soundfile as sf
from tqdm import tqdm

# -----------------------------
# Configuration
# -----------------------------
SAMPLE_RATE = 16000
N_MELS = 64
AUDIO_DURATION = 1  # seconds
MAX_LENGTH = SAMPLE_RATE * AUDIO_DURATION

# -----------------------------
# Input & Output Paths
# -----------------------------
BASE_DATA_DIR = "C:/Users/Admin/Desktop/Cockpit_voice_detection/data"

RAW_COMMAND = os.path.join(BASE_DATA_DIR, "raw", "raw_command")
RAW_TRIGGER = os.path.join(BASE_DATA_DIR, "raw", "raw_trigger")
AUG_COMMAND = os.path.join(BASE_DATA_DIR, "augmented", "augmented_command")
AUG_TRIGGER = os.path.join(BASE_DATA_DIR, "augmented", "augmented_trigger")
NOISE_DIR = os.path.join(BASE_DATA_DIR, "noise")
LABEL_COMMAND = os.path.join(BASE_DATA_DIR, "label", "command_labels.json")

FEATURES_DIR = os.path.join(BASE_DATA_DIR, "features")
VAD_COMMAND_FEATURES = os.path.join(FEATURES_DIR, "vad", "command")
VAD_TRIGGER_FEATURES = os.path.join(FEATURES_DIR, "vad", "trigger")
VAD_NOISE_FEATURES = os.path.join(FEATURES_DIR, "vad", "noise")

TRIGGER_COMMAND_FEATURES = os.path.join(FEATURES_DIR, "trigger", "command")
TRIGGER_TRIGGER_FEATURES = os.path.join(FEATURES_DIR, "trigger", "trigger")

ASR_COMMAND_FEATURES = os.path.join(FEATURES_DIR, "asr")

# -----------------------------
# Log-Mel Feature Extraction
# -----------------------------
def extract_log_mel(audio_path, sr=SAMPLE_RATE, n_mels=N_MELS):
    y, _ = librosa.load(audio_path, sr=sr)
    if len(y) < MAX_LENGTH:
        y = np.pad(y, (0, MAX_LENGTH - len(y)))
    else:
        y = y[:MAX_LENGTH]
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    log_mel = librosa.power_to_db(mel)
    return log_mel

# -----------------------------
# Generic Directory Preprocessing
# -----------------------------
def preprocess_directory(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for file in tqdm(os.listdir(input_dir), desc=f"Processing {os.path.basename(input_dir)}"):
        if file.endswith(".wav"):
            input_path = os.path.join(input_dir, file)
            output_path = os.path.join(output_dir, file.replace(".wav", ".npy"))
            log_mel = extract_log_mel(input_path)
            np.save(output_path, log_mel)

# -----------------------------
# ASR-Specific Preprocessing
# -----------------------------
def extract_asr_features(audio_dir, label_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    with open(label_file, 'r') as f:
        label_dict = json.load(f)

    for file in tqdm(os.listdir(audio_dir), desc=f"Processing ASR: {os.path.basename(audio_dir)}"):
        if file.endswith(".wav"):
            audio_path = os.path.join(audio_dir, file)
            log_mel = extract_log_mel(audio_path)
            np.save(os.path.join(output_dir, file.replace(".wav", ".npy")), log_mel)

            transcript = label_dict.get(file, "")
            with open(os.path.join(output_dir, file.replace(".wav", ".txt")), "w") as tf:
                tf.write(transcript)

# -----------------------------
# Full Preprocessing Pipeline
# -----------------------------
def preprocess_all():
    print("🔄 Starting preprocessing...")

    # VAD
    preprocess_directory(RAW_TRIGGER, os.path.join(VAD_TRIGGER_FEATURES, "raw"))
    preprocess_directory(AUG_TRIGGER, os.path.join(VAD_TRIGGER_FEATURES, "augmented"))
    preprocess_directory(NOISE_DIR, VAD_NOISE_FEATURES)

    # Trigger
    preprocess_directory(RAW_TRIGGER, os.path.join(TRIGGER_TRIGGER_FEATURES, "raw"))
    preprocess_directory(AUG_TRIGGER, os.path.join(TRIGGER_TRIGGER_FEATURES, "augmented"))

    # ASR
    extract_asr_features(RAW_COMMAND, LABEL_COMMAND, os.path.join(ASR_COMMAND_FEATURES, "raw"))
    extract_asr_features(AUG_COMMAND, LABEL_COMMAND, os.path.join(ASR_COMMAND_FEATURES, "augmented"))

    print("✅ Preprocessing done!")

# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    preprocess_all()
