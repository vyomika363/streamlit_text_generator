import torch
import torch.nn as nn

class MLPTextGenerator(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, context_size, activation="ReLU"):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc1 = nn.Linear(embed_dim * context_size, hidden_dim)

        # Choose activation function dynamically
        if activation == "ReLU":
            self.act = nn.ReLU()
        elif activation == "Tanh":
            self.act = nn.Tanh()
        elif activation == "Sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.fc2 = nn.Linear(hidden_dim, vocab_size)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.embedding(x).view(x.size(0), -1)
        x = self.act(self.fc1(x))
        x = self.log_softmax(self.fc2(x))
        return x
