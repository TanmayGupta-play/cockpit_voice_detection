import sys
import os
import numpy as np
from torch.utils.data import DataLoader

# Add project root to system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.datasetloader import ASRDataset, TriggerDataset, VADDataset

# -----------------------------
# 🔤 Test ASR Dataset
# -----------------------------
print("🔤 Testing ASR Dataset")
asr_dataset = ASRDataset(raw=True)  # use raw=False to test augmented data
asr_loader = DataLoader(asr_dataset, batch_size=2, shuffle=True)

for features, transcript in asr_loader:
    print("ASR Features Shape:", features.shape)
    print("ASR Transcript:", transcript)
    break

# -----------------------------
# 🔔 Test Trigger Dataset
# -----------------------------
print("\n🔔 Testing Trigger Dataset")
trigger_dataset = TriggerDataset()
trigger_loader = DataLoader(trigger_dataset, batch_size=2, shuffle=True)

for features, label in trigger_loader:
    print("Trigger Features Shape:", features.shape)
    print("Trigger Labels:", label)
    break

# -----------------------------
# 🎙️ Test VAD Dataset
# -----------------------------
print("\n🎙️ Testing VAD Dataset")
vad_dataset = VADDataset()
vad_loader = DataLoader(vad_dataset, batch_size=2, shuffle=True)

for features, label in vad_loader:
    print("VAD Features Shape:", features.shape)
    print("VAD Labels:", label)
    break