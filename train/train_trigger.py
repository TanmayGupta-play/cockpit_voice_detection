import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from models.trigger_word import TriggerWordModel
from utils.datasetloader import TriggerDataset
from sklearn.metrics import accuracy_score
from tqdm import tqdm

EPOCHS = 20
BATCH_SIZE = 32
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "checkpoints/trigger_word_model.pth"

dataset = TriggerDataset()
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = TriggerWordModel().to(DEVICE)
criterion = nn.BCELoss()
optimizer = Adam(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    all_preds, all_labels = [], []

    for features, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        features = features.unsqueeze(1).to(DEVICE)
        labels = labels.float().unsqueeze(1).to(DEVICE)

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

torch.save(model.state_dict(), CHECKPOINT_PATH)
print(f"\n💾 Model saved to: {CHECKPOINT_PATH}")