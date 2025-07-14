# models/trigger_word_model.py
import torch
import torch.nn as nn

class TriggerWordModel(nn.Module):
    def __init__(self):
        super(TriggerWordModel, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),  # [1, 16, 32, 16]

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # [1, 32, 16, 8]

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # [1, 64, 1, 1]
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = self.conv(x)
        return torch.sigmoid(self.classifier(x))  # keep sigmoid here for inference