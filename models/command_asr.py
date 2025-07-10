# models/command_asr.py
import torch
import torch.nn as nn

class CommandASRModel(nn.Module):
    def __init__(self, n_mels=64, hidden_dim=128, vocab_size=100):
        super(CommandASRModel, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
        )

        self.rnn = nn.GRU(input_size=32 * 32, hidden_size=hidden_dim,
                          num_layers=2, batch_first=True, bidirectional=True)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, vocab_size)
        )

    def forward(self, x):  # x: [B, 64, T]
        x = x.unsqueeze(1)         # [B, 1, 64, T]
        x = self.cnn(x)            # [B, 32, 32, T//2]
        x = x.permute(0, 3, 1, 2)  # [B, T//2, 32, 32]
        x = x.reshape(x.size(0), x.size(1), -1)  # [B, T//2, 32*32]
        output, _ = self.rnn(x)
        return self.classifier(output[:, -1, :])  # Only last timestep output