import torch
import torch.nn as nn

class TriggerCRNN(nn.Module):
    def __init__(self):
        super(TriggerCRNN, self).__init__()

        # CNN Layers
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),   # (B, 1, 64, T) -> (B, 16, 64, T)
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),                       # (B, 16, 32, T/2)

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # (B, 32, 32, T/2)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),                       # (B, 32, 16, T/4)
        )

        # Bi-GRU Layer
        self.gru = nn.GRU(
            input_size=32 * 16,  # channels * height
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(64 * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, 1, 64, T)
        x = self.cnn(x)                       # (B, 32, 16, T/4)
        x = x.permute(0, 3, 1, 2)             # (B, T/4, 32, 16)
        x = x.reshape(x.size(0), x.size(1), -1)  # (B, T/4, 512)
        x, _ = self.gru(x)                    # (B, T/4, 128)
        x = x[:, -1, :]                       # Take last time step (B, 128)
        out = self.classifier(x)             # (B, 1)
        return out.squeeze(1)                # (B,)