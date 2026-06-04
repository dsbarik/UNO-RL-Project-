import torch.nn as nn


class DQN(nn.Module):
    def __init__(self, input_dim=113, output_dim=61):
        super(DQN, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x):
        return self.net(x)
