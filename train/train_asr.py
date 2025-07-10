# train/train_asr.py
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from models.command_asr import CommandASRModel
from utils.datasetloader import ASRDataset
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import numpy as np

# -------------------------------
# Config
# -------------------------------
EPOCHS = 20
BATCH_SIZE = 16
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "checkpoints/command_asr.pth"

# -------------------------------
# Dataset & Label Encoder
# -------------------------------
dataset = ASRDataset(raw=True)
all_transcripts = [transcript for _, transcript in dataset]
char_set = sorted(set("".join(all_transcripts)))
char2idx = {c: i + 1 for i, c in enumerate(char_set)}  # leave 0 for blank
char2idx["<blank>"] = 0
idx2char = {i: c for c, i in char2idx.items()}

vocab_size = len(char2idx)

def text_to_indices(text):
    return [char2idx[c] for c in text if c in char2idx]

def collate_fn(batch):
    features, transcripts = zip(*batch)
    features = torch.tensor(np.stack(features), dtype=torch.float32)
    input_lengths = torch.full((len(features),), features.shape[2] // 2, dtype=torch.int32)  # T//2 after pooling

    labels = [torch.tensor(text_to_indices(t), dtype=torch.int32) for t in transcripts]
    label_lengths = torch.tensor([len(t) for t in labels], dtype=torch.int32)
    labels = torch.cat(labels)
    return features, labels, input_lengths, label_lengths

train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

# -------------------------------
# Model, Loss, Optimizer
# -------------------------------
model = CommandASRModel(vocab_size=vocab_size).to(DEVICE)
criterion = nn.CTCLoss(blank=0, zero_infinity=True)
optimizer = Adam(model.parameters(), lr=LR)

# -------------------------------
# Training Loop
# -------------------------------
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    for features, labels, input_lengths, label_lengths in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        features = features.to(DEVICE)  # [B, 64, T]
        labels = labels.to(DEVICE)
        input_lengths = input_lengths.to(DEVICE)
        label_lengths = label_lengths.to(DEVICE)

        logits = model(features)  # [B, T//2, vocab_size]
        logits = logits.permute(1, 0, 2)  # [T//2, B, vocab_size]

        loss = criterion(logits, labels, input_lengths, label_lengths)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"✅ Epoch {epoch+1}: CTC Loss = {epoch_loss:.4f}")

# -------------------------------
# Save Model
# -------------------------------
os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
torch.save({
    "model_state": model.state_dict(),
    "char2idx": char2idx,
    "idx2char": idx2char
}, CHECKPOINT_PATH)
print(f"\n💾 Model saved to: {CHECKPOINT_PATH}")