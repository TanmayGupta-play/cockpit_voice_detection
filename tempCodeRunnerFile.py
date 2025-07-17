import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from sklearn.metrics import accuracy_score
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.vad_crnn import VADCRNN
from utils.datasetloader import VADDataset

# -------------------------------
# Config
# -------------------------------
EPOCHS = 50
BATCH_SIZE = 32
LR = 1e-3
WEIGHT_DECAY = 1e-4
VALIDATION_SPLIT = 0.2
EARLY_STOPPING_PATIENCE = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "checkpoints/vad_crnn.pth"

# -------------------------------
# Dataset Split
# -------------------------------
full_dataset = VADDataset()
val_size = int(VALIDATION_SPLIT * len(full_dataset))
train_size = len(full_dataset) - val_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# -------------------------------
# Model, Loss, Optimizer
# -------------------------------
model = VADCRNN().to(DEVICE)
criterion = nn.BCELoss()
optimizer = Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# -------------------------------
# Evaluation Function
# -------------------------------
def evaluate(model, loader):
    model.eval()
    val_preds, val_labels = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.float().to(DEVICE)

            outputs = model(x)
            preds = (outputs > 0.5).float()

            val_preds += preds.cpu().numpy().astype(int).tolist()
            val_labels += y.cpu().numpy().astype(int).tolist()

    acc = accuracy_score(val_labels, val_preds)
    return acc

# -------------------------------
# Training Loop with Early Stopping
# -------------------------------
best_val_acc = 0.0
patience_counter = 0

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    all_preds, all_labels = [], []

    for features, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        features = features.to(DEVICE)  # [B, 1, 64, T]
        labels = labels.float().to(DEVICE)

        outputs = model(features).squeeze()
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        all_preds += (outputs.detach().cpu().numpy() > 0.5).astype(int).tolist()
        all_labels += labels.cpu().numpy().astype(int).tolist()

    train_acc = accuracy_score(all_labels, all_preds)
    val_acc = evaluate(model, val_loader)

    print(f"✅ Epoch {epoch+1}: Loss = {epoch_loss:.4f} | Train Acc = {train_acc:.4f} | Val Acc = {val_acc:.4f}")

    # Early Stopping
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), CHECKPOINT_PATH)
        print(f"💾 Best model saved with Val Acc = {val_acc:.4f}")
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print("⏹️ Early stopping triggered.")
            break