import torch
import torch.nn as nn

class ASRModel(nn.Module):
    def __init__(self, n_mels=64, hidden_size=128, vocab_size=30):
        super(ASRModel, self).__init__()

        self.n_mels = n_mels

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # keep input shape
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 1))  # ↓ reduce time dimension by 2 (T → T//2)
        )

        # Final CNN output shape: [B, 32, 64, T//2]
        self.gru_input_size = 16 * n_mels  # = 2048 if n_mels = 64

        self.gru = nn.GRU(
            input_size=self.gru_input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.classifier = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, x):  # x: [B, 1, 64, T]
        x = self.cnn(x)  # → [B, 32, 64, T//2]
        x = x.permute(0, 3, 1, 2)  # → [B, T//2, 32, 64]
        x = x.reshape(x.size(0), x.size(1), -1)  # → [B, T//2, 2048]
        x, _ = self.gru(x)  # → [B, T//2, 256 * 2]
        x = self.classifier(x)  # → [B, T//2, vocab_size]
        return x.log_softmax(2)  # for CTC loss

    def get_output_lengths(self, input_lengths):
        # Only time is downsampled (T → T//2)
        return input_lengths 
