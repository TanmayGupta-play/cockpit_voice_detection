# train_trigger_model.py
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

from models.trigger_word import TriggerWordModel  # Use the redefined model
from utils.datasetloader import TriggerDataset

# ---------------------
# Hyperparameters
# ---------------------
EPOCHS = 50
BATCH_SIZE = 32
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "checkpoints/trigger_word_model.pth"

# ---------------------
# Data Loading
# ---------------------
dataset = TriggerDataset()
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ---------------------
# Model, Loss, Optimizer
# ---------------------
model = TriggerWordModel().to(DEVICE)
criterion = nn.BCELoss()
optimizer = Adam(model.parameters(), lr=LR, weight_decay=1e-5)

# ---------------------
# Training Loop
# ---------------------
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0
    all_preds = []
    all_labels = []

    for features, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        features = features.unsqueeze(1).to(DEVICE)  # [B, 1, 64, T]
        labels = labels.float().unsqueeze(1).to(DEVICE)

        outputs = model(features)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        all_preds += (outputs.detach().cpu().numpy() > 0.5).astype(int).flatten().tolist()
        all_labels += labels.cpu().numpy().astype(int).flatten().tolist()

    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    print(f"\n✅ Epoch {epoch+1}/{EPOCHS}")
    print(f"   Loss     : {epoch_loss:.4f}")
    print(f"   Accuracy : {acc:.4f}")
    print(f"   Precision: {prec:.4f} | Recall: {rec:.4f} | F1-score: {f1:.4f}")

# ---------------------
# Save the Model
# ---------------------
torch.save(model.state_dict(), CHECKPOINT_PATH)
print(f"\n💾 Trigger model saved to: {CHECKPOINT_PATH}")