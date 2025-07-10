# train/train_vad.py

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from models.vad_crnn_model import VAD_CRNN
from utils.dataset_loader import VADDataset
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# -------------------------------
# Config
# -------------------------------
EPOCHS = 20
BATCH_SIZE = 32
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "checkpoints/vad_crnn.pth"

# -------------------------------
# Load Dataset
# -------------------------------
dataset = VADDataset()
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# -------------------------------
# Model, Loss, Optimizer
# -------------------------------
model = VAD_CRNN().to(DEVICE)
criterion = nn.BCELoss()
optimizer = Adam(model.parameters(), lr=LR)

# -------------------------------
# Training Loop
# -------------------------------
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    all_preds, all_labels = [], []

    for features, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        features = features.unsqueeze(1).to(DEVICE)  # [B, 1, 64, 32]
        labels = labels.float().unsqueeze(1).to(DEVICE)  # [B, 1]

        outputs = model(features)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        all_preds += (outputs.detach().cpu().numpy() > 0.5).astype(int).tolist()
        all_labels += labels.cpu().numpy().astype(int).tolist()

    acc = accuracy_score(all_labels, all_preds)
    print(f"✅ Epoch {epoch+1}: Loss = {epoch_loss:.4f} | Accuracy = {acc:.4f}")

# -------------------------------
# Save Model
# -------------------------------
torch.save(model.state_dict(), CHECKPOINT_PATH)
print(f"\n💾 Model saved to: {CHECKPOINT_PATH}")