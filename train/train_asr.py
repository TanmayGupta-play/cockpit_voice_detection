import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
from utils.datasetloader import ASRDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# ----------------------
# Load all transcripts first
# ----------------------
train_dataset = ASRDataset(raw=True)
all_transcripts = [transcript for _, transcript in train_dataset]

label_encoder = LabelEncoder()
label_encoder.fit(all_transcripts)

# ----------------------
# Collate Function
# ----------------------
def collate_fn(batch):
    features, transcripts = zip(*batch)
    features = torch.tensor(np.stack(features), dtype=torch.float32)
    labels = torch.tensor(label_encoder.transform(transcripts), dtype=torch.long)
    return features, labels

# ----------------------
# Config
# ----------------------
EPOCHS = 20
BATCH_SIZE = 32
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "checkpoints/asr_model.pth"

# ----------------------
# DataLoader
# ----------------------
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

# ----------------------
# Model
# ----------------------
class SimpleASRClassifier(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=128, output_dim=len(label_encoder.classes_)):
        super(SimpleASRClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(16 * 32 * 16, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # [B, 1, 64, 32]
        return self.net(x)

model = SimpleASRClassifier().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=LR)

# ----------------------
# Training Loop
# ----------------------
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for features, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        features = features.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(features)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    print(f"✅ Epoch {epoch+1}: Loss={total_loss:.4f} | Accuracy={acc:.4f}")

# ----------------------
# Save model
# ----------------------
torch.save(model.state_dict(), CHECKPOINT_PATH)
print(f"💾 ASR model saved at {CHECKPOINT_PATH}")