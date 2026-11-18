import torch.nn as nn


def conv_block(in_ch, out_ch, drop):
    """[Conv-BN-ReLU] x2 -> MaxPool -> Dropout2d"""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Dropout2d(drop),
    )


class ModifiedCNN(nn.Module):
    """
    Model 2 - CNN yang dimodifikasi dari SimpleCNN:
      1. Lebih dalam   : 4 blok konvolusi, tiap blok 2 layer Conv 3x3
      2. BatchNorm     : setelah setiap Conv -> training lebih stabil
      3. Dropout       : Dropout2d di tiap blok + Dropout 0.5 sebelum classifier
      4. Global Average Pooling : pengganti flatten besar -> parameter lebih sedikit
    """

    def __init__(self, input_c=3, output=50):
        super(ModifiedCNN, self).__init__()
        self.features = nn.Sequential(
            conv_block(input_c, 32, 0.10),  # 128 -> 64
            conv_block(32, 64, 0.15),       # 64  -> 32
            conv_block(64, 128, 0.20),      # 32  -> 16
            conv_block(128, 256, 0.25),     # 16  -> 8
        )
        self.pool = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, output),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))
