
import torch
import torch.nn as nn


class SimpleCNN(nn.Module):

    def __init__(self, input_dim=128, input_c=3, output=50):
        super(SimpleCNN, self).__init__()

        self.convBlock1 = nn.Sequential(
            nn.Conv2d(
                in_channels=input_c,
                out_channels=20,
                kernel_size=5
            ),
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3)
        )

        self.convBlock2 = nn.Sequential(
            nn.Conv2d(
                in_channels=20,
                out_channels=50,
                kernel_size=5
            ),
            nn.BatchNorm2d(50),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3)
        )

        # Mengurangi ukuran fitur sebelum masuk ke FC
        self.global_pool = nn.AdaptiveAvgPool2d((4, 4))

        # Mengurangi ketergantungan pada fitur tertentu
        self.dropout = nn.Dropout(p=0.5)

        self.fc = nn.Linear(
            in_features=50 * 4 * 4,
            out_features=output
        )

    def forward(self, x):

        x = self.convBlock1(x)
        x = self.convBlock2(x)

        x = self.global_pool(x)
        x = torch.flatten(x, 1)

        x = self.dropout(x)

        x = self.fc(x)

        return x
