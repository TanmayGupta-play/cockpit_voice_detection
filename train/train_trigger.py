import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.trigger_word import TriggerCRNN
from utils.datasetloader import TriggerDataset

# ---------------------
# Hyperparameters
# ---------------------
EPOCHS = 50
BATCH_SIZE = 32
LR = 1e-3
PATIENCE = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "checkpoints/trigger_word_model.pth"

# ---------------------
# Dataset & Dataloader
# ---------------------
full_dataset = TriggerDataset()
val_size = int(0.2 * len(full_dataset))
train_size = len(full_dataset) - val_size
train_set, val_set = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

# ---------------------
# Model, Loss, Optimizer
# ---------------------
model = TriggerCRNN().to(DEVICE)
criterion = nn.BCELoss()
optimizer = Adam(model.parameters(), lr=LR, weight_decay=1e-5)

# ---------------------
# Training Loop
# ---------------------
best_f1 = 0
patience_counter = 0

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    train_preds, train_labels = [], []

    for features, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
        features = features.unsqueeze(1).to(DEVICE)  # [B, 1, 64, T]
        labels = labels.float().to(DEVICE)

        outputs = model(features)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_preds += (outputs.detach().cpu().numpy() > 0.5).astype(int).tolist()
        train_labels += labels.cpu().numpy().astype(int).tolist()

    train_acc = accuracy_score(train_labels, train_preds)

    # ---------------------
    # Validation
    # ---------------------
    model.eval()
    val_loss = 0
    val_preds, val_labels = [], []

    with torch.no_grad():
        for features, labels in val_loader:
            features = features.unsqueeze(1).to(DEVICE)
            labels = labels.float().to(DEVICE)

            outputs = model(features)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            val_preds += (outputs.cpu().numpy() > 0.5).astype(int).tolist()
            val_labels += labels.cpu().numpy().astype(int).tolist()

    val_acc = accuracy_score(val_labels, val_preds)
    val_prec = precision_score(val_labels, val_preds, zero_division=0)
    val_rec = recall_score(val_labels, val_preds, zero_division=0)
    val_f1 = f1_score(val_labels, val_preds, zero_division=0)

    # ---------------------
    # Logs
    # ---------------------
    print(f"\n📊 Epoch {epoch+1}")
    print(f"   Train Loss : {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"   Val Loss   : {val_loss:.4f} | Val Acc  : {val_acc:.4f}")
    print(f"   Precision  : {val_prec:.4f} | Recall: {val_rec:.4f} | F1-score: {val_f1:.4f}")

    # ---------------------
    # Checkpoint + Early Stop
    # ---------------------
    if val_f1 > best_f1:
        best_f1 = val_f1
        torch.save(model.state_dict(), CHECKPOINT_PATH)
        print(f"💾 Best model saved with F1-score: {val_f1:.4f}")
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print("⏹️ Early stopping triggered.")
            break