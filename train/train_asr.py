import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from models.command_asr import CommandASRModel
from utils.datasetloader import ASRDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from tqdm import tqdm

EPOCHS = 20
BATCH_SIZE = 32
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "checkpoints/asr_model.pth"

# Load data
raw_dataset = ASRDataset(raw=True)
aug_dataset = ASRDataset(raw=False)
full_dataset = raw_dataset.data + aug_dataset.data

# Label encoding
texts = [trans for _, trans in full_dataset]
label_encoder = LabelEncoder()
label_encoder.fit(texts)
num_classes = len(label_encoder.classes_)

# Custom collate function
def collate_fn(batch):
    features, transcripts = zip(*batch)
    features = torch.tensor(np.stack(features), dtype=torch.float32)
    labels = torch.tensor(label_encoder.transform(transcripts))
    return features, labels

train_loader = DataLoader(raw_dataset + aug_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

model = CommandASRModel(vocab_size=num_classes).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    all_preds, all_labels = [], []

    for features, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        features = features.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(features)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        all_preds += preds.cpu().tolist()
        all_labels += labels.cpu().tolist()

    acc = accuracy_score(all_labels, all_preds)
    print(f"✅ Epoch {epoch+1}: Loss = {epoch_loss:.4f} | Accuracy = {acc:.4f}")

torch.save(model.state_dict(), CHECKPOINT_PATH)
print(f"\n💾 ASR Model saved to: {CHECKPOINT_PATH}")