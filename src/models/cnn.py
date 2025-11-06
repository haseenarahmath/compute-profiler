import torch.nn as nn

class TinyCNN(nn.Module):
    def __init__(self, in_ch=3, num_classes=10, width=32):
        super().__init__()
        c = width
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, c, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(c, 2*c, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(2*c, 2*c, 3, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.cls = nn.Linear(2*c, num_classes)

    def forward(self, x):
        h = self.features(x)
        h = h.flatten(1)
        return self.cls(h)
