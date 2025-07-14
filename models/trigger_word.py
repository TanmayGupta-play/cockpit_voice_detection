# models/trigger_word_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class TriggerWordModel(nn.Module):
    def __init__(self, n_mels=64, gru_hidden=64):
        super(TriggerWordModel, self).__init__()
        
        # CNN Encoder for feature extraction
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),   # [B, 16, 64, T]
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),                         # [B, 16, 32, T//2]

            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # [B, 32, 32, T//2]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))                          # [B, 32, 16, T//4]
        )

        # RNN expects input: [B, T, Features]
        self.bi_gru = nn.GRU(input_size=32 * 16, hidden_size=gru_hidden,
                             batch_first=True, bidirectional=True)

        # Attention Layer
        self.attention = nn.Linear(gru_hidden * 2, 1)

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(gru_hidden * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [B, 1, 64, T]
        x = self.cnn(x)  # -> [B, 32, 16, T//4]
        x = x.permute(0, 3, 1, 2)  # -> [B, T//4, 32, 16]
        x = x.flatten(start_dim=2)  # -> [B, T//4, 32*16]

        gru_out, _ = self.bi_gru(x)  # -> [B, T//4, 2*hidden]

        # Attention: weighted average of GRU outputs
        attn_weights = F.softmax(self.attention(gru_out), dim=1)  # [B, T, 1]
        context = torch.sum(attn_weights * gru_out, dim=1)        # [B, 2*hidden]

        out = self.classifier(context)  # [B, 1]
        return out