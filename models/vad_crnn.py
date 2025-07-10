import torch
import torch.nn as nn

class VADCRNN(nn.Module):
    def __init__(self, n_mels=64, hidden_size=64):
        super(VADCRNN, self).__init__()

        # CNN block for feature extraction
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # Input: [B, 1, 64, T]
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2))             # Output: [B, 16, 32, T//2]
        )

        # GRU for temporal modeling
        self.gru = nn.GRU(
            input_size=16 * 32,       # Flattened spatial features per timestep
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True
        )

        # Classifier for binary speech detection
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):  # Input x: [B, 64, T]
        x = x.unsqueeze(1)           # [B, 1, 64, T]
        x = self.cnn(x)              # [B, 16, 32, T//2]
        x = x.permute(0, 3, 1, 2)    # [B, T//2, 16, 32]
        x = x.reshape(x.size(0), x.size(1), -1)  # [B, T//2, 16*32]
        
        output, _ = self.gru(x)      # [B, T//2, hidden*2]
        out = output[:, -1, :]       # Take last timestep

        return self.classifier(out).squeeze(1)   # [B] → sigmoid output