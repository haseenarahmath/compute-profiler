import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim=128, hidden=256, out_dim=10, depth=3, dropout=0.0):
        super().__init__()
        layers, d = [], in_dim
        for _ in range(max(0, depth - 1)):
            layers += [nn.Linear(d, hidden), nn.ReLU(inplace=True)]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = hidden
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
